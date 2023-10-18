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
#include <iterator>
#include <map>
#include <memory>
#include <numeric>
#include <span>
#include <thread>
#include <type_traits>
#include <typeinfo>
#include <vector>

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
        double ms = duration * 0.001;
        return ms;
    }

   private:
    std::chrono::time_point<std::chrono::high_resolution_clock> start_time_point;
    std::chrono::time_point<std::chrono::high_resolution_clock> end_time_point;
};

enum class sch_t { CPU, GPU, MULTIGPU };

[[nodiscard]] sch_t get_sch_enum(std::string_view str) {
    static const std::map<std::string_view, sch_t> schmap = {
        {"cpu", sch_t::CPU}, {"gpu", sch_t::GPU}, {"multigpu", sch_t::MULTIGPU}};

    if (schmap.contains(str)) {
        return schmap.at(str);
    }

    throw std::invalid_argument("FATAL: " + std::string(str) +
                                " is not a stdexec scheduler.\n"
                                "Available schedulers: cpu (static thread pool), gpu, multigpu.\n"
                                "Exiting...\n");
}
