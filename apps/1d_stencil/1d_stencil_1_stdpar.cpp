//  Copyright (c) 2023 Weile Wei
//
// This example provides a stdpar implementation for the 1D stencil code.

#include "commons.hpp"
#include <boost/program_options.hpp>

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

            // TODO: nvc++ for_eaach_n lambda cannot capture static/global variable
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
int bencharmark(boost::program_options::variables_map& vm) {
   std::uint64_t nx =
        vm["nx"].as<std::uint64_t>();    // Number of grid points.
    std::uint64_t nt = vm["nt"].as<std::uint64_t>();    // Number of steps.

    if (vm.count("no-header"))
        header = false;

    // Create the stepper object
    stepper step;

    // Measure execution time.
    auto t = std::chrono::high_resolution_clock::now();

    // Execute nt time steps on nx grid points.
    stepper::space solution = step.do_work(nx, nt);

    // Print the final solution
    // TODO: make it default false
    if (!vm.count("results"))
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

    bencharmark(vm);

    return 0;
}
