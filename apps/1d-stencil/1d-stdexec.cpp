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
#if defined(USE_GPU)
#include <nvexec/multi_gpu_context.cuh>
#include <nvexec/stream_context.cuh>
#endif
#include <stdexec/execution.hpp>

#include "argparse/argparse.hpp"
#include "commons.hpp"
#include "repeat_n/repeat_n.cuh"

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
    std::string& sch = kwarg("sch",
                             "stdexec scheduler: [options: cpu"
#if defined(USE_GPU)
                             ", gpu, multigpu"
#endif  //USE_GPU
                             "]")
                           .set_default("cpu");

    int& nthreads = kwarg("nthreads", "number of threads").set_default(std::thread::hardware_concurrency());
};

using Real_t = double;
///////////////////////////////////////////////////////////////////////////////
// Command-line variables
constexpr Real_t k = 0.5;      // heat transfer coefficient
constexpr Real_t dt = 1.;      // time step
constexpr Real_t dx = 1.;      // grid spacing

///////////////////////////////////////////////////////////////////////////////
//[stepper_1
struct stepper {

    // do all the work on 'size' data points for 'nt' time steps
    [[nodiscard]] std::vector<Real_t> do_work(const auto& sch, std::size_t size, std::size_t nt) {
        std::vector<Real_t> current(size);
        std::vector<Real_t> next(size);

        Real_t **next_ptr_ptr = new Real_t *(next.data());
        Real_t **current_ptr_ptr = new Real_t*(current.data());

        stdexec::sender auto init = stdexec::bulk(stdexec::schedule(sch), size, [=](int i) { auto current_ptr = *current_ptr_ptr; ; current_ptr[i] = (Real_t)i; });
        stdexec::sync_wait(std::move(init));

#if !defined(USE_GPU)
        for (auto iter = 0; iter < nt; iter++)
#endif
        // evolve the system
        stdexec::sync_wait(
#if defined(USE_GPU)
            ex::just() | exec::on(sch,
           repeat_n(
             nt,
#else
            stdexec::schedule(sch) |
#endif
             stdexec::bulk(size, [=](int i) {
                    auto current_ptr = *current_ptr_ptr;
                    auto next_ptr = *next_ptr_ptr;

                    std::size_t left = (i == 0) ? size - 1 : i - 1;
                    std::size_t right = (i == size - 1) ? 0 : i + 1;
                    next_ptr[i] = current_ptr[i] + (k * dt / (dx * dx)) * (current_ptr[left] - 2 * current_ptr[i] + current_ptr[right]);
                })
            | stdexec::then([=]() { std::swap(*next_ptr_ptr, *current_ptr_ptr); })
#if defined(USE_GPU)
            ))
#endif // USE_GPU
        );

        if (nt % 2 == 0) {
            return current;
        }
        return next;
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
#if defined(USE_GPU)
            case sch_t::GPU:
                solution = step.do_work(nvexec::stream_context().get_scheduler(), size, nt);
                break;
            case sch_t::MULTIGPU:
                solution = step.do_work(nvexec::multi_gpu_stream_context().get_scheduler(), size, nt);
                break;
#endif  // USE_GPU
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
