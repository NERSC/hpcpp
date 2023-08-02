//  Copyright (c) 2023 Weile Wei
//
// This example provides a stdpar implementation for the 1D stencil code.

#include "commons.hpp"
#include "argparse/argparse.hpp"
#include <experimental/mdspan>

#include <stdexec/execution.hpp>
#include <exec/static_thread_pool.hpp>
#include <exec/any_sender_of.hpp>

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
    bool &time = kwarg("t, time", "print time").set_default(true);
};

///////////////////////////////////////////////////////////////////////////////
// Command-line variables
bool header = true;    // print csv heading
double k = 0.5;        // heat transfer coefficient
double dt = 1.;        // time step
double dx = 1.;        // grid spacing



template <class... Ts>
using any_sender_of = typename exec::any_receiver_ref<
    stdexec::completion_signatures<Ts...>>::template any_sender<>;

///////////////////////////////////////////////////////////////////////////////
//[stepper_1
struct stepper
{
    // Our partition type
    typedef double partition;

    // Our data for one time step
    typedef thrust::device_vector<partition> space;

    void init_value(auto& data, std::size_t np, std::size_t nx) {
        for(std::size_t i = 0; i != np; ++i) {
            double base_value = double(i * nx);
            for(std::size_t j = 0; j != nx; ++j) {
                data[i * nx + j] = base_value + double(j);
            }
        }
    }

    void init_zeros(auto& data, std::size_t np, std::size_t nx) {
        auto size = np * nx;
        for(std::size_t i = 0; i != size; ++i) {
            data[i] = double(0);
        }
    }

    // Our operator
    double heat(double left, double middle, double right, const double k = ::k, const double dt = ::dt, const double dx = ::dx)
    {
        return middle + (k * dt / (dx * dx)) * (left - 2 * middle + right);
    }

    inline std::size_t idx(std::size_t id, int dir, std::size_t size)
    {
        if (id == 0 && dir == -1) {
            return size - 1;
        }

        if (id == size - 1 && dir == +1) {
            return (std::size_t) 0; 
        }
        assert(id < size);

        return id + dir;
    }

    // do all the work on 'nx' data points for 'nt' time steps
    space do_work(stdexec::scheduler auto& sch, std::size_t np, std::size_t nx, std::size_t nt)
    {
        std::size_t size = np * nx;
        thrust::device_vector<partition> current_vec(size);
        thrust::device_vector<partition> next_vec(size);
        init_value(current_vec, np, nx);

        for (std::size_t t = 0; t != nt; ++t) {
            auto current_ptr = thrust::raw_pointer_cast(current_vec.data());
            auto next_ptr = thrust::raw_pointer_cast(next_vec.data());
            auto sender =
                stdexec::schedule(sch)
                | stdexec::bulk(np, [=, k= ::k, dt = ::dt, dx = ::dx, nx=nx, np=np](int i) {
                    for(int j = 0; j < nx; j++) {
                        std::size_t id = i * nx + j;
                        auto left = idx(id, -1, np * nx);
                        auto right = idx(id, +1, np * nx);
                        next_ptr[id] = heat(current_ptr[left], current_ptr[id], current_ptr[right], k, dt, dx);
                    }
                });
            stdexec::sync_wait(std::move(sender));
            current_vec.swap(next_vec);
        }

        return current_vec; 
    }
};

///////////////////////////////////////////////////////////////////////////////
int benchmark(args_params_t const & args) {
    std::uint64_t np = args.np;    // Number of partitions.
    std::uint64_t nx = args.nx;    // Number of grid points.
    std::uint64_t nt = args.nt;    // Number of steps.

    // Create the stepper object
    stepper step;

    nvexec::stream_context stream_ctx{};
    stdexec::scheduler auto sch = stream_ctx.get_scheduler();

    // Measure execution time.
    Timer timer;

    // Execute nt time steps on nx grid points.
    stepper::space solution = step.do_work(sch, np, nx, nt);

    auto time = timer.stop();

    // Print the final solution
    if (args.results)
    {
        for (std::size_t i = 0; i != np; ++i) {
            std::cout << "U[" << i << "] = {"; 
            for (std::size_t j = 0; j != nx; ++j) {
                std::cout << solution[i*nx + j] << " ";
            }
            std::cout << "}\n";
        }
    }

    if (args.time) {
        std::cout << "Duration: " << time << " ms." << "\n";
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