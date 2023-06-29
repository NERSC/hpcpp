//  Copyright (c) 2023 Weile Wei
//
// This example provides a sender implementation for the 1D stencil code.

#include "argparse/argparse.hpp"

#include <algorithm>
#include <iostream>
#include <stdexec/execution.hpp>
#include <exec/static_thread_pool.hpp>
#include <exec/any_sender_of.hpp>

// parameters
struct args_params_t : public argparse::Args
{
    bool &results = kwarg("results", "print generated results (default: false)").set_default(false);
    std::uint64_t &nx = kwarg("nx", "Local x dimension (of each partition)").set_default(100);
    std::uint64_t &nt = kwarg("nt", "Number of time steps").set_default(45);
    std::uint64_t &np = kwarg("np", "Number of partitions").set_default(10);
    bool &k = kwarg("k", "Heat transfer coefficient").set_default(0.5);
    double &dt = kwarg("dt", "Timestep unit (default: 1.0[s])").set_default(1.0);
    double &dx = kwarg("dx", "Local x dimension").set_default(1.0);
    bool &no_header = kwarg("no-header", "Do not print csv header row (default: false)").set_default(false); 
    bool &help = kwarg("h, help", "print help").set_default(false);
};

template <class... Ts>
using any_sender_of = typename exec::any_receiver_ref<
    stdexec::completion_signatures<Ts...>>::template any_sender<>;

///////////////////////////////////////////////////////////////////////////////
// Command-line variables
bool header = true;    // print csv heading
double k = 0.5;        // heat transfer coefficient
double dt = 1.;        // time step
double dx = 1.;        // grid spacing

///////////////////////////////////////////////////////////////////////////////
//[stepper_1
struct stepper
{
    // Our partition type
    typedef double partition;

    // Our data for one time step
    typedef std::vector<partition> space;

    using any_space_sender =
        any_sender_of<stdexec::set_value_t(space),
                      stdexec::set_stopped_t(),
                      stdexec::set_error_t(std::exception_ptr)>;

    // Our operator
    double heat(double left, double middle, double right, const double k = ::k, const double dt = ::dt, const double dx = ::dx)
    {
        return middle + (k * dt / (dx * dx)) * (left - 2 * middle + right);
    }

    // do all the work on 'nx' data points for 'nt' time steps
    auto do_work(std::size_t nx, std::size_t nt) -> any_space_sender
    {
        if (nt == 0) {
            space current(nx);
            for (std::size_t i = 0; i != nx; ++i)
                current[i] = double(i);
            return stdexec::just(current);
        }

        return stdexec::just(nt - 1)
            | stdexec::let_value([=](std::size_t nt_updated) { return do_work(nx, nt_updated); })
            | stdexec::then([=](auto current) { 

                space next(nx);

                next[0] = heat(current[nx - 1], current[0], current[1]);

                auto currentPtr = current.data();
                auto nextPtr = next.data();

                for (std::size_t i = 1; i != nx - 1; ++i)
                    next[i] = heat(current[i - 1], current[i], current[i + 1]);


                next[nx - 1] = heat(current[nx - 2], current[nx - 1], current[0]);

                return next;

            });
    }
};

///////////////////////////////////////////////////////////////////////////////
int benchmark(args_params_t const & args) {
    std::uint64_t nx = args.nx;    // Number of grid points.
    std::uint64_t nt = args.nt;    // Number of steps.

    // Create the stepper object
    stepper step;

    exec::static_thread_pool pool(8);
    stdexec::scheduler auto sch = pool.get_scheduler();
    stdexec::sender auto begin = stdexec::schedule(sch);

    // Measure execution time.
    auto t = std::chrono::high_resolution_clock::now();

    stdexec::sender auto sender =
        begin
        | stdexec::then([=]() { return nt; })
        | stdexec::let_value([=, &step](std::uint64_t nt) { return step.do_work(nx, nt); });

    auto [solution] = stdexec::sync_wait(std::move(sender)).value();

    // Print the final solution
    if (args.results)
    {
        for (std::size_t i = 0; i != nx; ++i)
            std::cout << "U[" << i << "] = " << solution[i] << std::endl;
    }

    auto elapsed = std::chrono::high_resolution_clock::now() - t;

    return 0;
}

int main(int argc, char* argv[])
{
    // parse params
    args_params_t args = argparse::parse<args_params_t>(argc, argv);
    // see if help wanted
    if (args.help)
    {
        args.print(); // prints all variables
        return 0;
    }

    benchmark(args);

    return 0;
}
