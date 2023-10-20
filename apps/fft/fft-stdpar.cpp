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

//
// fft algorithm
//
[[nodiscard]] std::vector<data_t> fft(const data_t *x, const int N, bool debug = false)
{
    std::vector<data_t> x_rev(N);
    std::vector<uint32_t> ind(N);

    // create mdspans
    auto x_r  = std::mdspan<data_t, view_1d, std::layout_right>(x_rev.data(), N);
    auto id   = std::mdspan<uint32_t, view_1d, std::layout_right>(ind.data(), N);

    // compute shift factor
    int shift = 32 - ilog2(N);

    // twiddle bits for fft
    std::for_each_n(std::execution::par_unseq, counting_iterator(0), N, [=](auto k){
        id(k) = reverse_bits32(k) >> shift;
        x_r(k) = x[id(k)];
    });

    // niterations
    int niters = ilog2(N);

    // local merge partition size
    int lN = 2;

    // set cout precision
    std::cout << std::fixed << std::setprecision(1);
    std::cout << "FFT progress: ";

    // iterate until niters - lN*=2 after each iteration
    for (int it = 0; it < niters; it++, lN*=2)
    {
        // print progress
        std::cout << (100.0 * it)/niters << "%.." << std::flush;

        // debugging timer
        static Timer dtimer;

        // number of partitions
        int nparts = N/lN;
        int tpp = lN/2;

        // display info only if debugging
        if (debug)
        {
            dtimer.start();
            std::cout << "lN = " << lN << ", npartitions = " << nparts << ", partition size = " << tpp << std::endl;
        }

        // parallel compute lN-pt FFT
        std::for_each_n(std::execution::par_unseq, counting_iterator(0), N, [=](auto k){
            // compute indices
            int  e   = (k/tpp)*lN + (k % tpp);
            auto o   = e + tpp;
            auto i   = (k % tpp);

            // compute 2-pt DFT
            auto tmp = x_r(e) + x_r(o) * WNk(N, i * nparts);
            x_r(o)     = x_r(e) - x_r(o) * WNk(N, i * nparts);
            x_r(e)     = tmp;
        });

        // print only if debugging
        if (debug)
            std::cout << "This iter time: " << dtimer.stop() << " ms" << std::endl;
    }

    // print final progress mark
    std::cout << "100%" << std::endl;

    // return x_rev = fft(x_r)
    return x_rev;
}

//
// simulation
//
int main(int argc, char* argv[])
{
    // parse params
    const fft_params_t args = argparse::parse<fft_params_t>(argc, argv);

    // see if help wanted
    if (args.help)
    {
        args.print();  // prints all variables
        return 0;
    }

    // simulation variables
    int N = args.N;
    sig_type_t sig_type = sig_type_t::box;
    //int freq = args.freq;
    bool print_sig = args.print_sig;
    bool print_time = args.print_time;
    bool validate = args.validate;

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

    // start the timer here
    Timer timer;

    // y[n] = fft(x[n])
    sig_t y_n(std::move(fft(x_n.data(), N, args.debug)));

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
        if (x_n.isFFT(y_n, exec::static_thread_pool(std::thread::hardware_concurrency()).get_scheduler()))
            std::cout << "SUCCESS: y[n] == fft(x[n])" << std::endl;
        else
            std::cout << "FAILED: y[n] != fft(x[n])" << std::endl;
    }

    return 0;
}
