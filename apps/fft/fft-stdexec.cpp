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
 * commons for the fft codes
 */

#define FFT_STDEXEC
#include "fft.hpp"
#include "repeat_n/repeat_n.cuh"

//
// fft algorithm
//
[[nodiscard]] std::vector<data_t> fft(const data_t* x, scheduler auto sch, const int N, const int max_threads,
                                      bool debug = false) {
    std::vector<data_t> x_rev(N);

    data_t* x_r = x_rev.data();

    // compute shift factor
    int shift = 32 - ilog2(N);

    // set cout precision
    fmt::print("FFT progress: ");

    // twiddle bits for fft
    ex::sender auto twiddle = ex::bulk(schedule(sch), N, [=](int k) {
        auto new_idx = reverse_bits32(k) >> shift;
        x_r[k] = x[new_idx];
    });
    ex::sync_wait(std::move(twiddle));

    // mark progress of the twiddle stage
    fmt::print("50%..");

    // niterations
    int niters = ilog2(N);

    // pointer to local partition size (must be dynamic mem to be copied to GPU)
    int* lN_ptr = new int(1);

#if !defined(USE_GPU)
    for (auto iter = 0; iter < niters; iter++)
#endif
        // evolve the system
        stdexec::sync_wait(
#if defined(USE_GPU)
            // iterate until niters - lN*=2 after each iteration
            ex::just() | exec::on(sch, repeat_n(niters,
#else
            stdexec::schedule(sch) |
#endif  // USE_GPU
                                                ex::then([=]() { *lN_ptr *= 2; }) |
                                                    ex::bulk(N / 2,
                                                             [=](int k) {
                                                                 // extract lN from pointer
                                                                 int lN = *lN_ptr;

                                                                 // number of partitions
                                                                 int nparts = N / lN;
                                                                 int tpp = lN / 2;

                                                                 // compute indices
                                                                 int e = (k / tpp) * lN + (k % tpp);
                                                                 auto o = e + tpp;
                                                                 auto i = (k % tpp);

                                                                 // compute 2-pt DFT
                                                                 auto tmp = x_r[e] + x_r[o] * WNk(N, i * nparts);
                                                                 x_r[o] = x_r[e] - x_r[o] * WNk(N, i * nparts);
                                                                 x_r[e] = tmp;
                                                             })
#if defined(USE_GPU)
                                                    ))
#endif  // USE_GPU
        );

    // print final progress mark
    fmt::print("100%\n");

    // return x_rev = fft(x_r)
    return x_rev;
}

//
// simulation
//
int main(int argc, char* argv[]) {
    // parse params
    const fft_params_t args = argparse::parse<fft_params_t>(argc, argv);

    // see if help wanted
    if (args.help) {
        args.print();  // prints all variables
        return 0;
    }

    // simulation variables
    int N = args.N;
    sig_type_t sig_type = sig_type_t::box;
    int max_threads = args.max_threads;
    //int freq = args.freq;
    bool print_sig = args.print_sig;
    bool print_time = args.print_time;
    bool validate = args.validate;
    std::string sched = args.sch;

    // x[n] signal
    sig_t x_n(N, sig_type);

    if (!isPowOf2(N)) {
        N = ceilPowOf2(N);
        fmt::print("INFO: N is not a power of 2. Padding zeros => N = {}\n", N);

        x_n.resize(N);
    }

    if (print_sig) {
        fmt::print("\nx[n] = ");
        x_n.printSignal();
    }

    // y[n] = fft(x[n]);
    std::vector<data_t> y(N);

    // start the timer here
    Timer timer;

    // initialize stdexec scheduler
    sch_t scheduler = get_sch_enum(sched);

    // launch with appropriate stdexec scheduler
    switch (scheduler) {
        case sch_t::CPU:
            y = fft(x_n.data(), exec::static_thread_pool(max_threads).get_scheduler(), N, max_threads, args.debug);
            break;
#if defined(USE_GPU)
        case sch_t::GPU:
            y = fft(x_n.data(), nvexec::stream_context().get_scheduler(), N, 1024 * 108, args.debug);
            break;
        case sch_t::MULTIGPU:
            y = fft(x_n.data(), nvexec::multi_gpu_stream_context().get_scheduler(), N, 4 * 1024 * 108, args.debug);
            break;
#endif  // USE_GPU
        default:
            throw std::runtime_error("Run: `fft-stdexec --help` to see the list of available schedulers");
    }

    // y[n] = fft(x[n])
    sig_t y_n(y);

    // stop timer
    auto elapsed = timer.stop();

    // print the fft(x)
    if (print_sig) {
        fmt::print("X(k) = ");
        y_n.printSignal();
    }

    // print the computation time
    if (print_time)
        fmt::print("Elapsed Time: {:f} ms\n", elapsed);

    // validate the recursively computed fft
    if (validate) {
        bool verify = true;
        // launch with appropriate stdexec scheduler
        switch (scheduler) {
            case sch_t::CPU:
                verify = x_n.isFFT(y_n, exec::static_thread_pool(max_threads).get_scheduler());
                break;
#if defined(USE_GPU)
            case sch_t::GPU:
                verify = x_n.isFFT(y_n, nvexec::stream_context().get_scheduler());
                break;
            case sch_t::MULTIGPU:
                verify = x_n.isFFT(y_n, nvexec::stream_context().get_scheduler());
                break;
#endif  // USE_GPU
            default:
                throw std::runtime_error("Run: `fft-stdexec --help` to see the list of available schedulers");
        }

        if (verify) {
            fmt::print("SUCCESS: y[n] == fft(x[n])\n");
        } else {
            fmt::print("FAILED: y[n] != fft(x[n])\n");
        }
    }

    return 0;
}
