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
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
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
#include <stdlib.h>
#include <stdint.h>
#include <vector>
#include <memory>
#include <iterator>
#include <typeinfo>
#include <type_traits>
#include <algorithm>
#include <iostream>
#include <execution>
#include <chrono>
#include <numeric>
#include <span>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include "counting_iterator.hpp"


// get mdpsan 2d indices from 1d index
#define dim2(x, ms)       int ii = x/ms.extent(1); int ij = x%ms.extent(1);
// get mdspan 3d indices from 1d index
#define dim3(x, ms)       int ii = x/(ms3.extent(1)*ms.extent(2)); int ij = (x/ms.extent(2))%ms.extent(1); int ik = x%ms.extent(2)
