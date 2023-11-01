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

#include "prefixSum.hpp"

//
// serial prefixSum function
//
template <typename T>
[[nodiscard]] data_t* prefixSum_stdpar(const T *in, const int N)
{
    data_t *y = new data_t[N];
    std::inclusive_scan(std::execution::par, in, in + N, y, std::plus<>());
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

    if (!isPowOf2(N))
    {
        N = ceilPowOf2(N);
        std::cout << "INFO: N != pow(2). Setting => N = " << N << std::endl;
    }

    // input data
    data_t *in = new data_t[N];

    std::cout << "Progress:0%" << std::flush;

    // random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<data_t> dist(1, 10);

    // fill random between 1 to 10
    std::generate(std::execution::seq, in, in+N, [&]() { return dist(gen); });

    std::cout << "..50%" << std::flush;

    // start the timer
    Timer timer;

    // stdpar prefixSum
    auto &&out = prefixSum_stdpar(in, N);

    // stop timer
    auto elapsed = timer.stop();

    std::cout << "..100%" << std::endl << std::flush;

    // print the input and its prefix sum (don't if N > 100)
    if (print_arr && N < 100)
    {
        std::cout << std::endl << "in  = ";
        printVec(in, N);

        std::cout << std::endl << "out = ";
        printVec(out, N);
        std::cout << std::endl;
    }

    // print the elapsed time
    if (print_time)
        std::cout << "Elapsed Time: " << elapsed << " s" << std::endl;

    // validate the prefixSum
    if (validate)
    {
        bool verify = psum::validatePrefixSum(in, out, N);

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
