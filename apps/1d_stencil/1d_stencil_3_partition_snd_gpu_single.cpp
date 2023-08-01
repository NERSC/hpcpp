//  Copyright (c) 2023 Weile Wei 

// This is the third in a series of examples demonstrating the development of a
// fully distributed solver for a simple 1D heat distribution problem.
//
// This example takes the code from example one and introduces a partitioning
// of the 1D grid into groups of grid partitions which are handled at the same time.
// The purpose is to be able to control the amount of work performed. The
// example is still fully serial, no parallelization is performed.

#include "commons.hpp"
#include "argparse/argparse.hpp"

#include <algorithm>
#include <stdexec/execution.hpp>
#include <exec/static_thread_pool.hpp>
#include <exec/any_sender_of.hpp>
#include <string>

#include <nvexec/stream_context.cuh>

#include <thrust/device_vector.h>

// parameters
struct args_params_t : public argparse::Args
{
    bool &results = kwarg("results", "print generated results (default: false)").set_default(false);
    std::uint64_t &nx = kwarg("nx", "Local x dimension (of each partition)").set_default(10);
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

inline std::size_t idx(std::size_t i, int dir, std::size_t size)
{
    if (i == 0 && dir == -1)
        return size - 1;
    if (i == size - 1 && dir == +1)
        return 0;

    assert((i + dir) < size);

    return i + dir;
}

///////////////////////////////////////////////////////////////////////////////
// Our partition_data data type
typedef thrust::device_vector<double> partition_data;


std::ostream& operator<<(std::ostream& os, partition_data const& c)
{
    os << "{";
    for (std::size_t i = 0; i != c.size(); ++i)
    {
        if (i != 0)
            os << ", ";
        os << c[i];
    }
    os << "}";
    return os;
}

///////////////////////////////////////////////////////////////////////////////
struct stepper
{
    // Our data for one time step
    typedef partition_data partition;
    typedef std::vector<partition> space;
    // typedef thrust::device_vector<partition> space;

    // Our operator
    static double heat(double left, double middle, double right, const double k, const double dt, const double dx)
    {
        return middle + (k * dt / (dx * dx)) * (left - 2 * middle + right);
    }

    // The partitioned operator, it invokes the heat operator above on all
    // elements of a partition.
    // static partition_data heat_part(partition_data& next, partition_data const& left,
    static void heat_part(partition_data& next, partition_data const& left,
        partition_data const& middle, partition_data const& right, const double k, const double dt, const double dx)
    {
        std::size_t size = middle.size();

        next[0] = heat(left[size - 1], middle[0], middle[1], k, dt, dx);

        for (std::size_t i = 1; i != size - 1; ++i)
            next[i] = heat(middle[i - 1], middle[i], middle[i + 1], k, dt, dx);

        next[size - 1] = heat(middle[size - 2], middle[size - 1], right[0], k, dt, dx);
    }

    void init_value(space& data) {
        for(std::size_t i = 0; i != data.size(); ++i) {
            std::size_t nx = data[i].size();
            double base_value = double(i * nx);
            for(std::size_t j = 0; j != nx; ++j) {
                data[i][j] = base_value + double(j);
            }
        }
    }

    // do all the work on 'np' partitions, 'nx' data points each, for 'nt'
    // time steps
    space do_work(stdexec::scheduler auto& sch, std::size_t np, std::size_t nx, std::size_t nt)
    {
        // Initial conditions: f(0, i) = i
        space current(np, partition_data(nx));
        space next(np, partition_data(nx));
        init_value(current);
        auto currentPtr = current.data();
        auto nextPtr = next.data();

        stdexec::sender auto sender =
            stdexec::schedule(sch) 
            | stdexec::bulk(np, [=, k= ::k, dt = ::dt, dx = ::dx](int i) {
                // Actual time step loop
                for (std::size_t t = 0; t != nt; ++t) {
                    heat_part(nextPtr[i], currentPtr[idx(i, -1, np)], currentPtr[i], currentPtr[idx(i, +1, np)],
                        k, dt, dx);
                    // swap current and next ptrs
                    currentPtr[i].swap(nextPtr[i]);
                }
            });

        stdexec::sync_wait(std::move(sender));
        return current; 
    }
};

///////////////////////////////////////////////////////////////////////////////
int benchmark(args_params_t const & args)
{
    std::uint64_t np = args.np;    // Number of partitions.
    std::uint64_t nx = args.nx;    // Number of grid points.
    std::uint64_t nt = args.nt;    // Number of steps.

    // Create the stepper object
    stepper step;

    // Measure execution time.
    auto t = std::chrono::high_resolution_clock::now();

    // Execute nt time steps on nx grid points and print the final solution.
    nvexec::stream_context stream_ctx{};
    stdexec::scheduler auto sch = stream_ctx.get_scheduler();
    stepper::space solution = step.do_work(sch, np, nx, nt);

    auto elapsed = std::chrono::high_resolution_clock::now() - t;

    // Print the final solution
    if (args.results)
    {
        for (std::size_t i = 0; i != nx; ++i) {
            std::cout << "U[" << i << "] = {"; 
            for (std::size_t j = 0; j != solution[i].size(); ++j) {
                std::cout << solution[i][j] << ", ";
            }
            std::cout << "}\n";
        }
    }

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
