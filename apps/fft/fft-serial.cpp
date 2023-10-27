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
// serial fft function
//
[[nodiscard]] std::vector<data_t> fft_serial(const data_t *x, const int N, bool debug = false)
{
    std::vector<data_t> x_r(N);

    // bit shift
    int shift = 32 - ilog2(N);

    // twiddle data in x[n]
    for (int k = 0; k < N; k++)
    {
        x_r[k] = x[reverse_bits32(k) >> shift];
    }

    // niterations
    int niters = ilog2(N);
    // local merge partition size
    int lN = 2;

    // set cout precision
    std::cout << std::fixed << std::setprecision(1);

    std::cout << "FFT progress: ";

    for (int k = 0; k < niters; k++, lN*=2)
    {
        std::cout << (100.0 * k)/niters << "%.." << std::flush;

        static Timer dtimer;

        // number of partitions
        int nparts = N/lN;
        int tpp = lN/2;

        if (debug)
            dtimer.start();

        // merge
        for (int k = 0; k < N/2; k++)
        {
            // compute indices
            int  e   = (k/tpp)*lN + (k % tpp);
            auto o   = e + tpp;
            auto i   = (k % tpp);
            auto tmp = x_r[e] + x_r[o] * WNk(N, i * nparts);
            x_r[o]     = x_r[e] - x_r[o] * WNk(N, i * nparts);
            x_r[e]     = tmp;
        }

        if (debug)
        std::cout << "This iter time: " << dtimer.stop() << " ms" << std::endl;
    }

    std::cout << "100%" << std::endl;
    return x_r;
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
    sig_type_t sig_type = getSignal(args.sig);
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

    // niterations
    int niters = ilog2(N);

    // start the timer
    Timer timer;

    // fft radix-2 algorithm
    // y[n] = fft(x[n]);
    sig_t y_n(std::move(fft_serial(x_n.data(), N, args.debug)));

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
