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

template <class... Ts>
using any_sender_of = typename exec::any_receiver_ref<
    stdexec::completion_signatures<Ts...>>::template any_sender<>;

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
struct partition_data
{
    partition_data(std::size_t size = 0)
      : data_(size)
    {
    }

    partition_data(std::size_t size, double initial_value)
      : data_(size)
    {
        double base_value = double(initial_value * size);
        for (std::size_t i = 0; i != size; ++i)
            data_[i] = base_value + double(i);
    }

    double& operator[](std::size_t idx)
    {
        return data_[idx];
    }
    double operator[](std::size_t idx) const
    {
        return data_[idx];
    }

    std::size_t size() const
    {
        return data_.size();
    }

    double* data()
    {
        return data_.data();
    }

private:
    std::vector<double> data_;
};

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
    space next;
    using any_space_sender =
    any_sender_of<stdexec::set_value_t(space),
                  stdexec::set_stopped_t(),
                  stdexec::set_error_t(std::exception_ptr)>;

    // Our operator
    static double heat(double left, double middle, double right, double k, double dt, double dx)
    {
        return middle + (k * dt / (dx * dx)) * (left - 2 * middle + right);
    }

    // The partitioned operator, it invokes the heat operator above on all
    // elements of a partition.
    static partition_data heat_part(partition_data const& left,
        partition_data & middle, partition_data const& right)
    {

        std::size_t size = middle.size();
        partition_data next(size);

        next[0] = heat(left[size - 1], middle[0], middle[1], k, dt, dx);

        auto nextPtr = next.data();
        auto middlePtr = middle.data();
        std::for_each_n(std::execution::par, counting_iterator(1), size-1,
            [=, k=k, dt=dt, dx=dx](int32_t i) {
            nextPtr[i] = heat(middlePtr[i - 1], middlePtr[i], middlePtr[i + 1], k, dt, dx);
        });

        next[size - 1] = heat(middle[size - 2], middle[size - 1], right[0], k, dt, dx);

        return next;
    }

    // do all the work on 'np' partitions, 'nx' data points each, for 'nt'
    // time steps
    auto do_work(std::size_t np, std::size_t nx, std::size_t nt) -> any_space_sender
    {
        if (nt == 0) {
            space current(np);
            next.resize(np);
            for (std::size_t i = 0; i != np; ++i)
                current[i] = partition_data(nx, double(i));
            return stdexec::just(current);
        }

        return stdexec::just(nt - 1)
            | stdexec::let_value([=](std::size_t nt_updated) { return do_work(np, nx, nt_updated); })
            | stdexec::bulk(np, [=](std::size_t i, auto current) { 
                next[i] = heat_part(current[idx(i, -1, np)], current[i], current[idx(i, +1, np)]);
            })
            | stdexec::then([&](auto current) { 
                return next;
        });
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

    exec::static_thread_pool pool(np);
    stdexec::scheduler auto sch = pool.get_scheduler();
    stdexec::sender auto begin = stdexec::schedule(sch);

    // Measure execution time.
    auto t = std::chrono::high_resolution_clock::now();

    stdexec::sender auto sender =
        begin
        | stdexec::then([=]() { return nt; })
        | stdexec::let_value([=, &step](std::uint64_t nt) { return step.do_work(np, nx, nt); });

    auto [solution] = stdexec::sync_wait(std::move(sender)).value();

    auto elapsed = std::chrono::high_resolution_clock::now() - t;

    // Print the final solution
    if (args.results)
    {
        for (std::size_t i = 0; i != nx; ++i)
            std::cout << "U[" << i << "] = " << solution[i] << std::endl;
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
