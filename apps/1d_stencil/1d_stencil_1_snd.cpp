//  Copyright (c) 2023 Weile Wei
//
// This example provides a sender implementation for the 1D stencil code.

#include <algorithm>
#include <iostream>
#include <stdexec/execution.hpp>
#include <exec/static_thread_pool.hpp>
#include <exec/any_sender_of.hpp>
#include <boost/program_options.hpp>

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
int benchmark(boost::program_options::variables_map& vm) {
   std::uint64_t nx =
        vm["nx"].as<std::uint64_t>();    // Number of grid points.
    std::uint64_t nt = vm["nt"].as<std::uint64_t>();    // Number of steps.

    if (vm.count("no-header"))
        header = false;

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
    // TODO: make it default false
    if (!vm.count("results"))
    {
        for (std::size_t i = 0; i != nx; ++i)
            std::cout << "U[" << i << "] = " << solution[i] << std::endl;
    }


    auto elapsed = std::chrono::high_resolution_clock::now() - t;

    return 0;
}

int main(int argc, char* argv[])
{
    namespace po = boost::program_options;

    // clang-format off
    po::options_description desc_commandline;
    desc_commandline.add_options()
        ("results", "print generated results (default: true)")
        ("nx", po::value<std::uint64_t>()->default_value(100),
         "Local x dimension")
        ("nt", po::value<std::uint64_t>()->default_value(45),
         "Number of time steps")
        ("k", po::value<double>(&k)->default_value(0.5),
         "Heat transfer coefficient (default: 0.5)")
        ("dt", po::value<double>(&dt)->default_value(1.0),
         "Timestep unit (default: 1.0[s])")
        ("dx", po::value<double>(&dx)->default_value(1.0),
         "Local x dimension")
        ( "no-header", "do not print out the csv header row")
    ;

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc_commandline), vm);
    po::notify(vm);

    benchmark(vm);

    return 0;
}
