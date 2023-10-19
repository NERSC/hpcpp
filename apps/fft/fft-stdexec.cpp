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

using namespace nvexec;

using any_void_sender =
      any_sender_of<stdexec::set_value_t(), stdexec::set_stopped_t(),
                    stdexec::set_error_t(std::exception_ptr)>;

//
// fft algorithm
//
std::vector<data_t> fft(scheduler auto sch, data_t *x, const int N, const int max_threads)
{
    std::vector<data_t> x_rev(N);
    std::vector<uint32_t> ind(N);

    data_t *x_r = x_rev.data();
    uint32_t *id = ind.data();

    int shift = 32 - ilog2(N);

    ex::sender auto twiddle = ex::transfer_just(sch, x_r, x, id)
        | ex::bulk(N, [=](int k, auto x_r, auto x, auto id){
            id[k] = reverse_bits32(k) >> shift;
            x_r[k] = x[id[k]];
        })
        | ex::then([](auto &&...){});

    ex::sync_wait(twiddle);

    // niterations
    int niters = ilog2(N);
    // local merge partition size
    int lN = 2;

    // set cout precision
    std::cout << std::fixed << std::setprecision(1);

    // transfer_just sender
    ex::sender auto tx = ex::transfer_just(sch, x_r);

    for (int k = 0; k < niters; k++, lN*=2)
    {
        std::cout << "FFT progress: " << (100.0 * k)/niters << "%" << std::endl;

        // number of partitions
        int stride = N/lN;

        if (lN < max_threads)
        {
            //std::cout << "lN = " << lN << ", partition size = " << stride << ", bulk = " << stride << ", each thread = " << lN/2 << std::endl;
            ex::sender auto merge = tx | ex::bulk(stride, [=](auto k, auto y)
            {
                // combine even and odd FFTs
                for (int i = 0; i < lN/2; i++)
                {
                    auto e = i + k*lN;
                    auto o = i + k*lN + lN/2;
                    auto tmp     = y[e] + y[o] * WNk(N, i * stride);
                    y[o] = y[e] - y[o] * WNk(N, i * stride);
                    y[e] = tmp;
                }
            });

            ex::sync_wait(std::move(merge));
        }
        else
        {
            //std::cout << "lN = " << lN << ", partition size = " << stride << ", bulk = " << lN/2 << ", x times called = " << stride << std::endl;
            // combine even and odd FFTs
            for (int i = 0; i < stride; i++)
            {
                ex::sender auto merge = tx | ex::bulk(lN/2, [=](auto k, auto y)
                {
                    auto e = k + i*lN;
                    auto o = k + i*lN + lN/2;
                    auto tmp = y[e] + y[o] * WNk(N, k * stride);
                    y[o] = y[e] - y[o] * WNk(N, k * stride);
                    y[e] = tmp;
                });

                ex::sync_wait(std::move(merge));
            }
        }
    }

    return x_rev;
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
    std::string sched = args.sch;

    // x[n] signal
    sig_t x_n(N, sig_type);

    if (!isPowOf2(N))
    {
        N = ceilPowOf2(N);
        std::cout << "INFO: N is not a power of 2. Padding zeros => N = " << N << std::endl;

        x_n.resize(N);
    }

    if (print_sig)
    {
        std::cout << std::endl << "x[n] = ";
        x_n.printSignal();
        std::cout << std::endl;
    }

    // y[n] = fft(x[n]);
    std::vector<data_t> y;

    // start the timer here
    Timer timer;

    // initialize stdexec scheduler
    sch_t scheduler = get_sch_enum(sched);

    // launch with appropriate stdexec scheduler
    switch (scheduler) {
        case sch_t::CPU:
            y = fft(exec::static_thread_pool(max_threads).get_scheduler(), x_n.data(), N, max_threads);
            break;
        case sch_t::GPU:
            y = fft(nvexec::stream_context().get_scheduler(), x_n.data(), N, 1024*108);
            break;
        case sch_t::MULTIGPU:
            y = fft(nvexec::multi_gpu_stream_context().get_scheduler(), x_n.data(), N, 4*1024*108);
            break;
        default:
            throw std::runtime_error("Run: `fft-stdexec --help` to see the list of available schedulers");
  }

    // y[n] = fft(x[n])
    sig_t y_n(y);

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
