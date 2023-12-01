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
[[nodiscard]] T* prefixSum(scheduler auto&& sch, const T* in, const int N) {
    // allocate a N+1 size array as there will be a trailing zero
    T* y = new T[N + 1];

    // number of iterations
    int niters = ilog2(N);

    // need to be dynamic memory to be able to use it in gpu ctx.
    int* d_ptr = new int(0);

    // memcpy to output vector to start computation.
    ex::sync_wait(ex::schedule(sch) | ex::bulk(N, [=](int k) { y[k] = in[k]; }));

    // GE Blelloch (1990) algorithm from pseudocode at:
    // https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda

    // upsweep
    for (int d = 0; d < niters; d++) {
        int bsize = N / (1 << d + 1);

        ex::sender auto uSweep = schedule(sch) | ex::bulk(bsize, [=](int k) {
                                     // stride1 = 2^(d+1)
                                     int st1 = 1 << d + 1;
                                     // stride2 = 2^d
                                     int st2 = 1 << d;
                                     // only the threads at indices (k+1) * 2^(d+1) -1 will compute
                                     int myIdx = (k + 1) * st1 - 1;

                                     // update y[myIdx]
                                     y[myIdx] += y[myIdx - st2];
                                 });
        // wait for upsweep
        ex::sync_wait(uSweep);
    }

    // write sum to y[N] and reset vars
    ex::sync_wait(schedule(sch) | ex::then([=]() {
                      y[N] = y[N - 1];
                      y[N - 1] = 0;
                  }));

    // downsweep
    for (int d = niters - 1; d >= 0; d--) {
        int bsize = N / (1 << d + 1);

        ex::sender auto dSweep = schedule(sch) | ex::bulk(bsize, [=](int k) {
                                     // stride1 = 2^(d+1)
                                     int st1 = 1 << d + 1;
                                     // stride2 = 2^d
                                     int st2 = 1 << d;
                                     // only the threads at indices (k+1) * 2^(d+1) -1 will compute
                                     int myIdx = (k + 1) * st1 - 1;

                                     // update y[myIdx] and y[myIdx-stride2]
                                     auto tmp = y[myIdx];
                                     y[myIdx] += y[myIdx - st2];
                                     y[myIdx - st2] = tmp;
                                 });

        // wait for downsweep
        ex::sync_wait(dSweep);
    }

    // return the computed results.
    return y;
}

//
// simulation
//
int main(int argc, char* argv[]) {
    // parse params
    const prefixSum_params_t args = argparse::parse<prefixSum_params_t>(argc, argv);

    // see if help wanted
    if (args.help) {
        args.print();  // prints all variables
        return 0;
    }

    // simulation variables
    int N = args.N;
    bool print_arr = args.print_arr;
    bool print_time = args.print_time;
    bool validate = args.validate;
    std::string sched = args.sch;
    int nthreads = args.nthreads;

    if (!isPowOf2(N)) {
        N = ceilPowOf2(N);
        fmt::print("INFO: N != pow(2). Setting => N = {}\n", N);
    }

    // input data
    data_t* in = new data_t[N];

    fmt::print("Progress:0%");

    // random number generator
    psum::genRandomVector(in, N, (data_t)0, (data_t)10);

    fmt::print("..50%");

    // output pointer
    data_t* out = nullptr;

    // start the timer
    Timer timer;

    // initialize stdexec scheduler
    sch_t scheduler = get_sch_enum(sched);

    // launch with appropriate stdexec scheduler
    switch (scheduler) {
        case sch_t::CPU:
            out = prefixSum(exec::static_thread_pool(nthreads).get_scheduler(), in, N);
            break;
#if defined(USE_GPU)
        case sch_t::GPU:
            out = prefixSum(nvexec::stream_context().get_scheduler(), in, N);
            break;
        case sch_t::MULTIGPU:
            out = prefixSum(nvexec::multi_gpu_stream_context().get_scheduler(), in, N);
            break;
#endif  // USE_GPU
        default:
            throw std::runtime_error("Run: `prefixSum-stdexec --help` to see the list of available schedulers");
    }

    // stop timer
    auto elapsed = timer.stop();

    fmt::print("..100%\n");

    // print the input and its prefix sum (don't if N > 100)
    if (print_arr && N < 100) {
        fmt::print("int = {}\n", fmt::join(in, in + N, " "));
        fmt::print("out = {}\n", fmt::join(out + 1, out + 1 + N, " "));
    }

    // print the elapsed time
    if (print_time)
        fmt::print("Elapsed Time: {:f} s\n", elapsed);

    // validate the prefixSum
    if (validate) {
        bool verify = psum::validatePrefixSum(in, out + 1, N);

        if (verify)
            fmt::print("SUCCESS..");
        else
            fmt::print("FAILED..");

        fmt::print("\n");
    }

    // return status
    return 0;
}
