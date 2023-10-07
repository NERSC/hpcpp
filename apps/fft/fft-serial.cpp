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
    sig_type_t sig_type = args.sig;
    int freq = args.freq;
    bool print_sig = args.print_sig;
    bool print_time = args.print_time;

    // x[n] signal
    //std::vector<data_t> test_sig{2,1,-1,5,0,3,0,-4};
    //N = test_sig.size();

    Timer timer;

    sig_t x_n(N, sig_type);

    if (!isPowOf2(N))
    {
        N = ceilPowOf2(N);
        std::cout << "log_2(N) != integer. Padding zeros for N = " << N << std::endl;

        x_n.resize(N);
    }

    sig_t y_n(x_n);

    if (print_sig)
    {
        std::cout << std::endl << "x[n] = ";
        x_n.printSignal();
        std::cout << std::endl;
    }

    // niterations
    int niters = ilog2(N);

    std::function<void(data_t *, int, const int)> fft = [&](data_t *x, int lN, const int N)
    {
        int stride = N/lN;

        if (lN == 2)
        {
            auto x_0 = x[0] + x[1]* WNk(N, 0);
            x[1] = x[0] - x[1]* WNk(N, 0);
            x[0] = x_0;
            return;
        }

        // vectors for left and right
        std::vector<data_t> e(lN/2);
        std::vector<data_t> o(lN/2);

        // copy data into vectors
        for (auto k = 0; k < lN/2; k++)
        {
            e[k] = x[2*k];
            o[k] = x[2*k+1];
        }

        // compute N/2 pt FFT on even
        fft(e.data(), lN/2, N);

        // compute N/2 pt FFT on odd
        fft(o.data(), lN/2, N);

        // combine even and odd FFTs
        for (int k = 0; k < lN/2; k++)
        {
            x[k] = e[k] + o[k] * WNk(N, k * stride);
            x[k+lN/2] = e[k] - o[k] * WNk(N, k * stride);
        }

        return;
    };

    // fft radix-2 algorithm with senders
    fft(y_n.data(), N, N);

    if (print_sig)
    {
        std::cout << "X[k] = ";
        y_n.printSignal();
        std::cout << std::endl;
    }

    auto elapsed = timer.stop();

    if (print_time)
        std::cout << "Elapsed Time: " << elapsed << " ms" << std::endl;

    return 0;
}