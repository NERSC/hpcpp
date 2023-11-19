/*
 * MIT License
 *
 * Copyright (c) 2023 The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of any
 * required approvals from the U.S. Dept. of Energy).  All rights reserved.
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

#pragma once

#include <math.h>
#include <stdint.h>
#include <stdlib.h>

#include <algorithm>
#include <bit>
#include <cassert>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <execution>
#include <functional>
#include <iostream>
#include <random>
#include <iterator>
#include <map>
#include <memory>
#include <numeric>
#include <span>
#include <thread>
#include <type_traits>
#include <typeinfo>
#include <vector>

#include <fmt/core.h>
#include <fmt/ranges.h>

#include <mdspan_formatter.hpp>

#include "counting_iterator.hpp"

// get mdpsan 2d indices from 1d index
#define dim2(x, ms)            \
    int ii = x / ms.extent(1); \
    int ij = x % ms.extent(1);
// get mdspan 3d indices from 1d index
#define dim3(x, ms)                              \
    int ii = x / (ms3.extent(1) * ms.extent(2)); \
    int ij = (x / ms.extent(2)) % ms.extent(1);  \
    int ik = x % ms.extent(2)

class Timer {
   public:
    Timer() { start(); }

    ~Timer() { stop(); }

    void start() { start_time_point = std::chrono::high_resolution_clock::now(); }

    double stop() {
        end_time_point = std::chrono::high_resolution_clock::now();
        return duration();
    }

    double duration() {
        auto start =
            std::chrono::time_point_cast<std::chrono::microseconds>(start_time_point).time_since_epoch().count();
        auto end = std::chrono::time_point_cast<std::chrono::microseconds>(end_time_point).time_since_epoch().count();
        auto duration = end - start;
        double ms = duration * 1e-6;
        return ms;
    }

   private:
    std::chrono::time_point<std::chrono::high_resolution_clock> start_time_point;
    std::chrono::time_point<std::chrono::high_resolution_clock> end_time_point;
};

enum class sch_t { CPU, GPU, MULTIGPU };

[[nodiscard]] sch_t get_sch_enum(std::string_view str) {
    static const std::map<std::string_view, sch_t> schmap = {
        {"cpu", sch_t::CPU},
#if defined (USE_GPU)
        {"gpu", sch_t::GPU}, {"multigpu", sch_t::MULTIGPU}
#endif // USE_GPU
};

    if (schmap.contains(str)) {
        return schmap.at(str);
    }

    throw std::invalid_argument("FATAL: " + std::string(str) +
                                " is not a stdexec scheduler.\n"
                                "Available schedulers: cpu"
#if defined (USE_GPU)
                                ", gpu, multigpu"
#endif
                                "\n"
                                "Exiting...\n");
}

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

inline int ceilPowOf2(unsigned int v)
{
  return static_cast<int>(std::bit_ceil(v));
}

inline int ilog2(uint32_t x)
{
    return static_cast<int>(log2(x));
}
template <typename T>
bool complex_compare(T a, T b, double error = 0.0101)
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

// alias for status variables
using status_t = int;
