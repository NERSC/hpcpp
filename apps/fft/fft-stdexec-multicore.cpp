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

#include "fft.hpp"

using any_void_sender =
      any_sender_of<stdexec::set_value_t(), stdexec::set_stopped_t(),
                    stdexec::set_error_t(std::exception_ptr)>;

//
// recursive multicore fft
//
any_void_sender fft_multicore(sender auto &&snd, data_t *x, int lN, const int N, int max_threads)
{
    // current merge stride
    int stride = N/lN;

    // if parallelism > max threads => serial
    if (stride >= max_threads)
    {
        // TODO: can this be improved? Putting it in ex::then doesn't sync
        fft_serial(x, lN, N);
        return just();
    }

    // base case
    if (lN == 2)
    {
        // TODO: can this be improved? Putting it in ex::then doesn't sync
        auto x_0 = x[0] + x[1]* WNk(N, 0);
        x[1] = x[0] - x[1]* WNk(N, 0);
        x[0] = x_0;

        return just();
    }

    // vectors for even and odd index elements
    std::vector<data_t> e(lN/2);
    std::vector<data_t> o(lN/2);

    // copy even and odd indexes to vectors and split sender
    ex::sender auto fork =
        ex::bulk(snd, lN/2, [&](int k){
            // copy data into vectors
            e[k] = x[2*k];
            o[k] = x[2*k+1];
        })
        | ex::split();

    // local thread pool and scheduler
    exec::static_thread_pool pool{std::min(lN/2, max_threads)};
    scheduler auto sched = pool.get_scheduler();

    // compute forked fft and merge
    ex::sender auto merge = when_all(
        fork | ex::then([=,&e](){
            // compute N/2 pt FFT on even

            // WHY: deadlock for N>=64 if `snd` here instead of schedule(sched) or just()
            // passing `fork` here results in compiler error (nvc++-Fatal-/path/to/tools/cpp1
            // TERMINATED by signal 11 - NVC++ 23.7 goes in forever loop)
            fft_multicore(schedule(sched), e.data(), lN/2, N, max_threads);
        }),
        fork | ex::then([=,&o](){
            // compute N/2 pt FFT on odd - same behavior
            fft_multicore(schedule(sched), o.data(), lN/2, N, max_threads);
        })
    )
    | ex::bulk(lN/2, [&](int k){
        // combine even and odd FFTs
        x[k] = e[k] + o[k] * WNk(N, k * stride);
        x[k+lN/2] = e[k] - o[k] * WNk(N, k * stride);
    });

    // wait to complete
    ex::sync_wait(merge);

    // return void sender
    return just();

}

//
// simulation
//
int main(int argc, char* argv[])
{
    // parse params
    fft_params_t args = argparse::parse<fft_params_t>(argc, argv);

    // see if help wanted
    if (args.help)
    {
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

    // start the timer
    Timer timer;

    // x[n] signal
    sig_t x_n(N, sig_type);

    if (!isPowOf2(N))
    {
        N = ceilPowOf2(N);
        std::cout << "INFO: N is not a power of 2. Padding zeros => N = " << N << std::endl;

        x_n.resize(N);
    }

    // y[n] = fft(x[n]);
    sig_t y_n(x_n);

    if (print_sig)
    {
        std::cout << std::endl << "x[n] = ";
        x_n.printSignal();
        std::cout << std::endl;
    }

    // niterations
    int niters = ilog2(N);

    // thread pool and scheduler
    exec::static_thread_pool pool{max_threads};
    scheduler auto sched = pool.get_scheduler();

    // fft radix-2 algorithm
    fft_multicore(schedule(sched), y_n.data(), N, N, max_threads);

    // stop timer
    auto elapsed = timer.stop();

    // print the fft(x)
    if (print_sig)
    {
        std::cout << "X(k) = ";
        y_n.printSignal();
        std::cout << std::endl;
    }

    // print the computation time
    if (print_time)
        std::cout << "Elapsed Time: " << elapsed << " ms" << std::endl;

    // validate the recursively computed fft
    if (validate)
    {
        if (x_n.isFFT(y_n))
            std::cout << "SUCCESS: y[n] == fft(x[n])" << std::endl;
        else
            std::cout << "FAILED: y[n] != fft(x[n])" << std::endl;
    }

    return 0;
}
