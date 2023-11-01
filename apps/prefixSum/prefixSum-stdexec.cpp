/*
 * MIT License
 *
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

/*
 * commons for the prefixSum codes
 */

#define PSUM_STDEXEC
#include "prefixSum.hpp"
#include "repeat_n/repeat_n.cuh"

//
// stdexec prefixSum function
//
template <typename T>
[[nodiscard]] ull_t* prefixSum(scheduler auto &&sch, const T *in, const int N)
{
    ull_t *y = new ull_t[N+1];

    // memcpy to output vector
    ex::sync_wait(ex::schedule(sch) | ex::bulk(N, [=](int k){ y[k] = in[k]; }));

    int niters = ceilPowOf2(N);
    int *d_ptr = new int(0);

    // GE Blelloch (1990) algorithm
    ex::sender auto pSum = ex::just()
    // upsweep algorithm
    | exec::on(sch,
        repeat_n(niters, ex::bulk(N/2, [=](int k){
            int d = *d_ptr;
            int s1 = 1 << d+1;
            int s2 = 1 << d;
            int my = (k+1) * s1 - 1;

            // only update the participating indices
            if (my < N)
                y[my] += y[my-s2];
        })
        | ex::then([=](){ *d_ptr += 1; })
    )
    // write last element to zero
    | ex::then([=](){
        y[N] = y[N-1];
        y[N-1] = 0;
        *d_ptr = niters-1;
    })
    // downsweep algorithm
    | exec::on(sch,
        repeat_n(niters, ex::bulk(N/2, [=](int k){
            int d = *d_ptr;
            int s1 = 1 << d+1;
            int s2 = 1 << d;
            int my = (k+1) * s1 - 1;

            // only update the participating indices
            if (my < N)
            {
                auto tmp = y[my];
                y[my] += y[my-s2];
                y[my-s2] = tmp;
            }

        })
        | ex::then([=](){ *d_ptr -= 1; })
    )
    ));

    ex::sync_wait(pSum);

    return y;
}

//
// simulation
//
int main(int argc, char* argv[])
{
    // parse params
    const prefixSum_params_t args = argparse::parse<prefixSum_params_t>(argc, argv);

    // see if help wanted
    if (args.help)
    {
        args.print();  // prints all variables
        return 0;
    }

    // simulation variables
    int N = args.N;
    bool print_arr = args.print_arr;
    bool print_time = args.print_time;
    bool validate = args.validate;
    std::string sched = args.sch;
    int max_threads = args.max_threads;

    // input data
    data_t *in = new data_t[N];

    // random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<data_t> dist(1, 10);

    // fill random between 1 to 10
    std::generate(std::execution::seq, in, in+N, [&]() { return dist(gen); });

    // start the timer
    Timer timer;

    ull_t *out = nullptr;

    // initialize stdexec scheduler
    sch_t scheduler = get_sch_enum(sched);

    // launch with appropriate stdexec scheduler
    switch (scheduler) {
        case sch_t::CPU:
            out = prefixSum(exec::static_thread_pool(max_threads).get_scheduler(), in, N);
            break;
#if defined(USE_GPU)
        case sch_t::GPU:
            out = prefixSum(nvexec::stream_context().get_scheduler(), in, N);
            break;
        case sch_t::MULTIGPU:
            out = prefixSum(nvexec::multi_gpu_stream_context().get_scheduler(), in, N);
            break;
#endif // USE_GPU
        default:
            throw std::runtime_error("Run: `prefixSum-stdexec --help` to see the list of available schedulers");
  }

    // stop timer
    auto elapsed = timer.stop();

    // print the input and its prefix sum (don't if N > 100)
    if (print_arr && N < 100)
    {
        std::cout << std::endl << "in  = ";
        printVec(in, N);
        std::cout << std::endl << "out = ";
        auto optr = out + 1;
        printVec(optr, N);
        std::cout << std::endl;
    }

    // print the elapsed time
    if (print_time)
        std::cout << "Elapsed Time: " << elapsed << " s" << std::endl;

    // validate the prefixSum
    if (validate)
    {
        bool verify = psum::validatePrefixSum(in, out+1, N);

        if (verify)
            std::cout << "SUCCESS..";
        else
            std::cout << "FAILED..";

        std::cout << std::endl;
    }

    delete[] in;
    delete[] out;

    return 0;
}
