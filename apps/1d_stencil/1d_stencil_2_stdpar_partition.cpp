//  Copyright (c) 2023 Weile Wei
//
// This example provides a stdpar implementation for the 1D stencil code.

#include "commons.hpp"
#include "argparse/argparse.hpp"
#include <experimental/mdspan>


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

inline std::size_t idx(std::size_t id, int dir, std::size_t size)
{
    if (id == 0 && dir == -1) {
        return size - 1;
    }
        
    if (id == size - 1 && (int) dir == +1) {
        return 0; 
    }

    assert(id < size);

    return id + dir;
}

///////////////////////////////////////////////////////////////////////////////
//[stepper_1
struct stepper
{
    // Our partition type
    typedef double partition;

    // Our data for one time step
    using view_1d = std::extents<int, std::dynamic_extent>;
    typedef std::mdspan<partition, view_1d, std::layout_right> space;

    void init_value(auto& data) {
        for(std::size_t i = 0; i != data.size(); ++i) {
            std::size_t size = data.size();
            double base_value = double(i * size);
            for(std::size_t j = 0; j != size; ++j) {
                data[i * size + j] = base_value + double(j);
            }
        }
    }

    // Our operator
    double heat(double left, double middle, double right, const double k = ::k, const double dt = ::dt, const double dx = ::dx)
    {
        return middle + (k * dt / (dx * dx)) * (left - 2 * middle + right);
    }

    // do all the work on 'nx' data points for 'nt' time steps
    space do_work(std::size_t np, std::size_t nx, std::size_t nt)
    {
        partition *current_ptr = new partition[np * nx];
        partition *next_ptr = new partition[np * nx];
        auto current = space (current_ptr, np*nx);
        auto next = space (next_ptr, np*nx);

        init_value(current);

        // Actual time step loop
        for (std::size_t t = 0; t != nt; ++t)
        {
            for(std::size_t i = 0; i < np; ++i) {
                std::for_each_n(std::execution::par, counting_iterator(0), nx,
                    [=, k=k, dt=dt, dx=dx](int32_t j) {
                    auto id = i * nx + j;
                    auto left = idx(id, -1, np * nx);
                    auto right = idx(id, +1, np * nx);
                    next[id] = heat(current[left], current[id], current[right], k, dt, dx);
                });
            }
            std::swap(current, next);
        }

        return current;
    }
};

///////////////////////////////////////////////////////////////////////////////
int benchmark(args_params_t const & args) {
    std::uint64_t np = args.np;    // Number of partitions.
    std::uint64_t nx = args.nx;    // Number of grid points.
    std::uint64_t nt = args.nt;    // Number of steps.

    // Create the stepper object
    stepper step;

    // Measure execution time.
    auto t = std::chrono::high_resolution_clock::now();

    // Execute nt time steps on nx grid points.
    auto solution = step.do_work(np, nx, nt);

    // Print the final solution
    if (args.results)
    {
        for (std::size_t i = 0; i != np; ++i) {
            std::cout << "U[" << i << "] = {"; 
            for (std::size_t j = 0; j != nx; ++j) {
                std::cout << solution[i*np + j] << " ";
            }
            std::cout << "}\n";
        }
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
