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

#include <bit>
#include <complex>
#include <functional>
#include <experimental/mdspan>
#include <exec/any_sender_of.hpp>
#include <exec/static_thread_pool.hpp>
#include <stdexec/execution.hpp>

#include "argparse/argparse.hpp"
#include "commons.hpp"

namespace ex = stdexec;
using namespace std::complex_literals;

// data type
using Real_t = double;
using data_t = std::complex<Real_t>;

enum class sig_type { square, sinusoid, sawtooth, triangle, sinc, box };
using sig_type_t = sig_type;

// fft radix
constexpr int radix = 2;

// parameters
struct fft_params_t : public argparse::Args {
  sig_type_t& sig = kwarg("sig", "input signal type: square, sinusoid, sawtooth, triangle, box").set_default(sig_type_t::box);
  int& freq = kwarg("f,freq", "Signal frequency").set_default(1024);
  int& N = kwarg("N", "N-point FFT").set_default(1024);
  bool& print_sig = flag("p,print", "print x[n] and X(k)");

#if defined(USE_OMP)
  int& nthreads = kwarg("nthreads", "number of threads").set_default(1);
#endif  // USE_OMP

  bool& help = flag("h, help", "print help");
  bool& print_time = flag("t,time", "print fft time");
};

inline bool isPowOf2(long long int x) {
  return !(x == 0) && !(x & (x - 1));
}

template <typename T>
void printVec(T &vec, int len)
{
    std::cout << "[ ";
    for (int i = 0; i < len; i++)
      std::cout << vec[i] << " ";

    std::cout << "]" << std::endl;
}

inline std::complex<Real_t> WNk(int N, int k)
{
    return std::complex<Real_t>(exp(-2*M_PI*1/N*k*1i));
}

inline int ceilPowOf2(unsigned int v)
{
  return static_cast<int>(std::bit_ceil(v));
}

inline int ilog2(uint32_t x)
{
    return static_cast<int>(log2(x));
}

class signal
{
public:

  signal() = default;
  signal(int N)
  {
    if (N <= 0)
    {
      std::cerr << "FATAL: N must be > 0. exiting.." << std::endl;
      exit(1);
    }
    y.reserve(ceilPowOf2(N));
    y.resize(N);
  }

  signal(signal &rhs)
  {
    y = rhs.y;
  }
  signal(std::vector<data_t> &in)
  {
    y = std::move(in);
  }

  signal(int N, sig_type type)
  {
    if (N <= 0)
    {
      std::cerr << "FATAL: N must be > 0. exiting.." << std::endl;
      exit(1);
    }
    y.reserve(ceilPowOf2(N));
    y.resize(N);
    signalGenerator(type);
  }

  void signalGenerator(sig_type type=sig_type::box)
  {
    int N = y.size();

    switch (type) {
      case sig_type::square:
        for (int n = 0; n < N; ++n)
          y[n] = (n < N / 4 || n > 3 * N/4) ? 1.0 : -1.0;
        break;
      case sig_type::sinusoid:
        for (int n = 0; n < N; ++n)
          y[n] = std::sin(2.0 * M_PI * n / N);
        break;
      case sig_type::sawtooth:
        for (int n = 0; n < N; ++n)
          y[n] = 2.0 * (n / N) - 1.0;
        break;
      case sig_type::triangle:
        for (int n = 0; n < N; ++n)
          y[n] = 2.0 * std::abs(2.0 * (n / N) - 1.0) - 1.0;
        break;
      case sig_type::sinc:
          y[0] = 1.0;
        for (int n = 1; n < N; ++n)
          y[n] = std::sin(2.0 * M_PI * n / N) / (2.0 * M_PI * n / N);
        break;
      case sig_type::box:
        for (int n = 0; n < N; ++n)
          y[n] = (n < N / 4 || n > 3 * N / 4) ? 1.0 : 0.0;
        break;
      default:
        std::cerr << "FATAL: Unknown signal type. exiting.." << std::endl;
        exit(1);
    }
  }

  ~signal()
  {
    y.clear();
  }

  data_t *data() { return y.data(); }
  int len() { return y.size(); }

  void resize(int N)
  {
    if (N != y.size())
      y.resize(N, 0);
  }

  data_t &operator[](int n)
  {
    return y[n];
  }

  data_t &operator()(int n)
  {
    return y[n];
  }

  void printSignal() {
    std::cout << std::fixed << std::setprecision(2);

    std::cout << "[ ";
    for (auto &el : y)
      std::cout << el << " ";

    std::cout << "]" << std::endl;
  }

private:
  // y[n]
  std::vector<data_t> y;
};

using sig_t = signal;
