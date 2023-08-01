//  Copyright (c) 2023 Weile Wei
//
// This example provides a stdpar implementation for the 1D stencil code.

#include "commons.hpp"
#include "argparse/argparse.hpp"


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

    // Our operator
    double heat(double left, double middle, double right, const double k = ::k, const double dt = ::dt, const double dx = ::dx)
    {
        return middle + (k * dt / (dx * dx)) * (left - 2 * middle + right);
    }

    // do all the work on 'nx' data points for 'nt' time steps
    space do_work(std::size_t nx, std::size_t nt)
    {
        // U[t][i] is the state of position i at time t.
        std::vector<space> U(2, space(nx));

        // Initial conditions: f(0, i) = i
        for (std::size_t i = 0; i != nx; ++i)
            U[0][i] = double(i);

        // Actual time step loop
        for (std::size_t t = 0; t != nt; ++t)
        {
            auto const & current = U[t % 2];
            auto& next = U[(t + 1) % 2];

            next[0] = heat(current[nx - 1], current[0], current[1]);

            auto currentPtr = current.data();
            auto nextPtr = next.data();

            std::for_each_n(std::execution::par_unseq, counting_iterator(1), nx-1,
                [=, k=k, dt=dt, dx=dx](int32_t i) {
                nextPtr[i] = heat(currentPtr[i - 1], currentPtr[i], currentPtr[i + 1], k, dt, dx);
            });


            next[nx - 1] = heat(current[nx - 2], current[nx - 1], current[0]);
        }

        // Return the solution at time-step 'nt'.
        return U[nt % 2];
    }
};

///////////////////////////////////////////////////////////////////////////////
int benchmark(args_params_t const & args) {
    std::uint64_t nx = args.nx;    // Number of grid points.
    std::uint64_t nt = args.nt;    // Number of steps.

    // Create the stepper object
    stepper step;

    // Measure execution time.
    auto t = std::chrono::high_resolution_clock::now();

    // Execute nt time steps on nx grid points.
    stepper::space solution = step.do_work(nx, nt);

    // Print the final solution
    if (args.results)
    {
        for (std::size_t i = 0; i != nx; ++i)
            std::cout << "U[" << i << "] = " << solution[i] << std::endl;
    }


    auto elapsed = std::chrono::high_resolution_clock::now() - t;

    // std::uint64_t const os_thread_count = hpx::get_os_thread_count();
    // TODO: print time results
    // print_time_results(os_thread_count, elapsed, nx, nt, header);

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
