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

#pragma once

#include <exec/static_thread_pool.hpp>
#include <stdexec/execution.hpp>

#if defined(USE_GPU)
#include <nvexec/multi_gpu_context.cuh>
#include <nvexec/stream_context.cuh>
using namespace nvexec;
#endif  //USE_GPU

#include "argparse/argparse.hpp"

#include "commons.hpp"

using namespace std;
using namespace stdexec;
namespace ex = stdexec;

// data type
using data_t = unsigned long long;

// input arguments
struct prefixSum_params_t : public argparse::Args {
    int& N = kwarg("N", "array size").set_default(1e9);
    bool& print_arr = flag("p,print", "print array and prefixSum");
    int& nthreads = kwarg("nthreads", "number of threads").set_default(std::thread::hardware_concurrency());

#if defined(PSUM_STDEXEC)
    std::string& sch = kwarg("sch",
                             "stdexec scheduler: [options: cpu"
#if defined(USE_GPU)
                             ", gpu, multigpu"
#endif  //USE_GPU
                             "]")
                           .set_default("cpu");
#endif  // PSUM_STDEXEC

    bool& validate = flag("validate", "validate the results");
    bool& help = flag("h, help", "print help");
    bool& print_time = flag("t,time", "print prefixSum time");
    bool& debug = flag("d,debug", "print internal timers and configs (if any)");
};

namespace psum {
template <typename T>
[[nodiscard]] bool validatePrefixSum(T* in, data_t* out, size_t N) {
    fmt::print("Validating: \n");

    // compute inclusive_scan via parSTL
    std::vector<data_t> test(N);
    std::inclusive_scan(std::execution::par, in, in + N, test.begin(), std::plus<>());

    // check if equal
    return std::equal(std::execution::par, out, out + N, test.begin());
}

template <typename T>
void genRandomVector(T* in, int N, T lower, T upper) {
    // random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<T> dist(lower, upper);

    // fill random between 1 to 10
    std::generate(std::execution::seq, in, in + N, [&gen, &dist]() { return dist(gen); });
}
}  // namespace psum
