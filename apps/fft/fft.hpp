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

#include <experimental/mdspan>
#include <complex>

#include "argparse/argparse.hpp"
#include "commons.hpp"

using namespace std::complex_literals;

// data type
using Real_t = double;
using data_t = std::complex<Real_t>;

// number of dimensions
constexpr int dims = 1;

// 1D view
using view_1d = std::extents<int, std::dynamic_extent>;

// 2D view
using view_2d = std::extents<int, std::dynamic_extent, std::dynamic_extent>;

// 3D view
using view_3d = std::extents<int, std::dynamic_extent, std::dynamic_extent,
                             std::dynamic_extent>;

enum class fft_type { fftw, cufft };
enum class sig_type { square, sinusoid, sawtooth, triangle, sinc, box };

using sig_type_t = sig_type;

// parameters
struct fft_params_t : public argparse::Args {
  sig_type_t& sig = kwarg("sig", "input signal type: square, sinusoid, sawtooth, triangle, box").set_default(sig_type_t::box);
  int& freq = kwarg("f,freq", "Signal frequency").set_default(1000);
  int& len = kwarg("n,N", "N-point FFT").set_default(1<<16);
  bool& print_fft = flag("p,print", "print Fourier transformed signal");

#if defined(USE_OMP)
  int& nthreads = kwarg("nthreads", "number of threads").set_default(1);
#endif  // USE_OMP

  bool& help = flag("h, help", "print help");
  bool& print_time = flag("t,time", "print transform time");
};

void printSignal(data_t* sig, int N) {
  std::cout << std::fixed << std::setprecision(1);

  for (int i = 0; i < N; ++i)
    std::cout << sig[i] << " ";

  std::cout << std::endl;
}

class signal
{
public:

  signal()
  {
    this->N = 1e3;
    t.resize(this->N);
    y.resize(this->N);
    dt = 1.0 / this->N;
  }

  signal(int _N)
  {
    if (_N <= 0)
    {
      std::cerr << "FATAL: N must be greater than 0. exiting.." << std::endl;
      exit(1);
    }
    this->N = _N;
    t.resize(this->N);
    y.resize(this->N);
    dt = 1.0 / this->N;
  }

  signal(int N, sig_type type=sig_type::box)
  {
    if (N <= 0)
    {
      std::cerr << "FATAL: N must be greater than 0. exiting.." << std::endl;
      exit(1);
    }

    this->N = N;
    t.resize(N);
    y.resize(N);
    dt = 1.0 / N;
    signalGenerator(N, type);
  }

  void signalGenerator(int N, sig_type type=sig_type::box)
  {
    int interval = 1/N;
    std::vector<Real_t> t(N);

    switch (type) {
      case sig_type::square:
        for (int i = 0; i < N; ++i)
          y[i] = (i < N / 4 || i > 3 * N/4) ? 1.0 : -1.0;
        break;
      case sig_type::sinusoid:
        for (int i = 0; i < N; ++i)
          y[i] = std::sin(2.0 * M_PI * i / N);
        break;
      case sig_type::sawtooth:
        for (int i = 0; i < N; ++i)
          y[i] = 2.0 * (i / N) - 1.0;
        break;
      case sig_type::triangle:
        for (int i = 0; i < N; ++i)
          y[i] = 2.0 * std::abs(2.0 * (i / N) - 1.0) - 1.0;
        break;
      case sig_type::sinc:
          y[0] = 1.0;
        for (int i = 1; i < N; ++i)
          y[i] = std::sin(2.0 * M_PI * i / N) / (2.0 * M_PI * i / N);
        break;
      case sig_type::box:
        for (int i = 0; i < N; ++i)
          y[i] = (i < N / 4 || i > 3 * N / 4) ? 1.0 : 0.0;
        break;
      default:
        std::cerr << "FATAL: Unknown signal type. exiting.." << std::endl;
        exit(1);
    }
  }

  ~signal()
  {
    y.clear();
    t.clear();
  }

private:
  int N;
  Real_t dt;
  // time axis
  std::vector<Real_t> t;
  // y(t) axis
  std::vector<Real_t> y;
};