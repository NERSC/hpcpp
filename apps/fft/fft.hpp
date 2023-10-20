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

#include <complex>

#include <experimental/mdspan>
#include <stdexec/execution.hpp>
#include <exec/static_thread_pool.hpp>

#if defined(GPUSTDPAR)
  #include <nvexec/stream_context.cuh>
  #include <nvexec/multi_gpu_context.cuh>
using namespace nvexec;
#endif //GPUSTDPAR

#include <experimental/linalg>
#include "argparse/argparse.hpp"

#include "commons.hpp"

using namespace std;
using namespace stdexec;
using namespace std::complex_literals;
using stdexec::sync_wait;

namespace ex = stdexec;

// 2D view
using view_2d = std::extents<int, std::dynamic_extent, std::dynamic_extent>;

// data type
using Real_t = double;
using data_t = std::complex<Real_t>;

// enum for signal types
enum sig_type { square, sinusoid, sawtooth, triangle, sinc, box };
using sig_type_t = sig_type;

// map for signals
std::map<std::string, sig_type_t> sigmap{{"square",sig_type_t::square}, {"sinusoid", sig_type_t::sinusoid}, {"triangle", sig_type_t::sawtooth},
                           {"triangle", sig_type_t::triangle}, {"sinc", sig_type_t::sinc}, {"box", sig_type_t::box}};

// custom get sig_type_t from string
sig_type_t getSignal(std::string &sig)
{
    if (sigmap.contains(sig))
    {
      return sigmap[sig];
    }
    else
    {
      return (sig_type_t)(-1);
    }
}

// input arguments
struct fft_params_t : public argparse::Args {
  // NVC++ is not supported by magic_enum so using strings
  std::string& sig = kwarg("sig", "input signal type: square, sinusoid, sawtooth, triangle, box").set_default("box");

  int& freq = kwarg("f,freq", "Signal frequency").set_default(1024);
  int& N = kwarg("N", "N-point FFT").set_default(1024);
  bool& print_sig = flag("p,print", "print x[n] and X(k)");
  int& max_threads = kwarg("nthreads", "number of threads").set_default(std::thread::hardware_concurrency());

#if defined(FFT_STDEXEC)
  std::string& sch = kwarg("sch", "stdexec scheduler: [options: cpu"
  #if defined (GPUSTDPAR)
                          ", gpu, multigpu"
  #endif //GPUSTDPAR
                          "]").set_default("cpu");
#endif  // FFT_STDEXEC

  bool& validate = flag("validate", "validate the results via y[k] = WNk * x[n]");
  bool& help = flag("h, help", "print help");
  bool& print_time = flag("t,time", "print fft time");
  bool& debug = flag("d,debug", "print internal timers and launch configs");
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

bool complex_compare(data_t a, data_t b, double error = 0.0101)
{
  auto r = (fabs(a.real() - b.real()) < error)? true: false;
  return r && (fabs(a.imag() - b.imag()) < error)? true: false;
}

uint32_t reverse_bits32(uint32_t x)
{
    x = ((x & 0xaaaaaaaa) >> 1) | ((x & 0x55555555) << 1);
    x = ((x & 0xcccccccc) >> 2) | ((x & 0x33333333) << 2);
    x = ((x & 0xf0f0f0f0) >> 4) | ((x & 0x0f0f0f0f) << 4);
    x = ((x & 0xff00ff00) >> 8) | ((x & 0x00ff00ff) << 8);
    return (x >> 16) | (x << 16);
}

class signal
{
public:

  signal() = default;
  signal(int N)
  {
    if (N <= 0)
    {
      std::cerr << "ERROR: N must be > 0. exiting.." << std::endl;
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

  signal(int N, sig_type type, int threads = std::thread::hardware_concurrency())
  {
    if (N <= 0)
    {
      std::cerr << "ERROR: N must be > 0. exiting.." << std::endl;
      exit(1);
    }
    y.reserve(ceilPowOf2(N));
    y.resize(N);
    signalGenerator(type, threads);
  }

  void signalGenerator(sig_type type=sig_type::box, int threads = std::thread::hardware_concurrency())
  {
    int N = y.size();

    // scheduler from a thread pool
    exec::static_thread_pool ctx{threads};
    scheduler auto sch = ctx.get_scheduler();

    // start scheduling
    sender auto start = schedule(sch);

    // generate input signal
    switch (type) {
      case sig_type::square:
        sync_wait(bulk(start, N, [&](int n) {
          for (int n = 0; n < N; ++n)
            y[n] = (n < N / 4 || n >= 3 * N/4) ? 1.0 : -1.0;
        }));
        break;
      case sig_type::sinusoid:
        sync_wait(bulk(start, N, [&](int n) {
          y[n] = std::sin(2.0 * M_PI * n / N);
        }));
        break;
      case sig_type::sawtooth:
        sync_wait(bulk(start, N, [&](int n) {
          y[n] = 2.0 * (n / N) - 1.0;
        }));
        break;
      case sig_type::triangle:
        sync_wait(bulk(start, N, [&](int n) {
          y[n] = 2.0 * std::abs(2.0 * (n / N) - 1.0) - 1.0;
        }));
        break;
      case sig_type::sinc:
          y[0] = 1.0;
          sync_wait(bulk(start, N-1, [&](int n) {
            y[n+1] = std::sin(2.0 * M_PI * (n+1) / N) / (2.0 * M_PI * (n+1) / N);
          }));
        break;
      case sig_type::box:
        sync_wait(bulk(start, N, [&](int n) {
          y[n] = (n < N / 4 || n >= 3 * N / 4) ? 1.0 : 0.0;
        }));
        break;
      default:
        std::cerr << "ERROR: Unknown input signal type. exiting.." << std::endl;
        std::cerr << "Run: <FFT_app> --help to see the list of available signals" << std::endl;
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

  bool isFFT(signal &X, scheduler auto sch, int maxN = 20000)
  {
    int N = y.size();
    bool ret = true;

    //int nparts = N/maxN;
    //int psize = std::min(N, nparts);
    //int matsize = psize * psize;

    std::vector<data_t> Y(N);
    std::vector<data_t> M(N*N);

    auto A   = std::mdspan<data_t, view_2d, std::layout_right>(M.data(), N, N);
    auto mdy = std::mdspan<data_t, view_2d, std::layout_right>(y.data(), N, 1);
    auto mdY = std::mdspan<data_t, view_2d, std::layout_right>(Y.data(), N, 1);

    data_t *F = M.data();
    data_t *X_ptr = X.data();
    data_t *Y_ptr = Y.data();

    ex::sender auto init = ex::transfer_just(sch, F) | ex::bulk(N*N, [=](int k, auto F){
      int i = k / N;
      int j = k % N;
      F[k] = WNk(N, i*j);
    });

    // initialize
    ex::sync_wait(init);

    // compute Y[n] = dft(x[n]) = WNk * x[n]
    stdex::linalg::matrix_product(std::execution::par, A, mdy, mdY);

    // compare the computed Y[n] (dft) with X[n](fft)
    ex::sender auto verify = ex::transfer_just(sch, ret, X_ptr, Y_ptr)
    | ex::bulk(N, [](int k, auto &ret, auto X_ptr, auto Y_ptr){
      if (!complex_compare(X_ptr[k], Y_ptr[k]))
      {
        //std::cout << "y[" << i << "] = " << X[i] << " != x[" << i << "] = " << Y[i] << std::endl;
        ret = false;
      }
    })
    | then([](auto ret, auto &&...)
    {
      return ret;
    });

    // let the pipeline run
    auto [re] = ex::sync_wait(verify).value();

    return re;
  }
private:
  // y[n]
  std::vector<data_t> y;
};

using sig_t = signal;