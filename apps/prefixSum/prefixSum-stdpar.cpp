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
[[nodiscard]] T* prefixSum_stdpar(const T* in, const int N) {
    T* y = new T[N];
    std::inclusive_scan(std::execution::par, in, in + N, y, std::plus<>());
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

    // serial prefixSum
    out = prefixSum_stdpar(in, N);

    // stop timer
    auto elapsed = timer.stop();

    fmt::print("..100%\n");

    // print the input and its prefix sum (don't if N > 100)
    if (print_arr && N < 100) {
        fmt::print("int = {}\n", fmt::join(in, in + N, " "));
        fmt::print("out = {}\n", fmt::join(out, out + N, " "));
    }

    // print the elapsed time
    if (print_time)
        fmt::print("Elapsed Time: {:f} s\n", elapsed);

    // validate the prefixSum
    if (validate) {
        bool verify = psum::validatePrefixSum(in, out, N);

        if (verify)
            fmt::print("SUCCESS..");
        else
            fmt::print("FAILED..");

        fmt::print("\n");
    }

    // delete in and out
    delete[] in;
    delete[] out;

    // return status
    return 0;
}
