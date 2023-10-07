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

#include <experimental/mdspan>

#include "commons.hpp"

using data_type = int;
// 2D view
using extents_type =
    std::extents<int, std::dynamic_extent, std::dynamic_extent>;
// 3D view (fix the first dimension to 2)
using extents_type2 =
    std::extents<int, 2, std::dynamic_extent, std::dynamic_extent>;

int main() {
  constexpr int N = 1e9;
  std::vector<data_type> v(N);

  // View data as contiguous memory representing 2 rows of 6 ints each
  auto ms2 = std::mdspan<data_type, extents_type, std::layout_right>(v.data(),
                                                                     N / 2, 2);
  // View the same data as a 3D array 2 (fixed above) x 3 x 2
  auto ms3 = std::mdspan<data_type, extents_type2, std::layout_right>(v.data(),
                                                                      N / 4, 2);

  // auto dim2 = [=](int i){int i1 = i/ms2.extent(1); int i2 = i%ms2.extent(1);
  // return std::make_tuple(i1, i2);}; auto dim3 = [=](int i){int i1 =
  // i/(ms3.extent(1)*ms3.extent(2)); int i2 = (i/ms3.extent(2))%ms3.extent(1);
  // int i3 = i%ms3.extent(2); return std::make_tuple(i1, i2, i3);};

  std::for_each(std::execution::par_unseq, ms2.data_handle(),
                ms2.data_handle() + ms2.size(), [=](int& i) {
                  auto global_idx = std::distance(ms2.data_handle(), &i);
                  dim2(global_idx, ms2);
                  // auto [i1, i2] = dim2(global_idx);
                  ms2(ii, ij) = global_idx;
                });

  std::cout << std::endl << std::endl;

  std::for_each(std::execution::par_unseq, ms2.data_handle(),
                ms2.data_handle() + ms2.size(), [=](int& i) {
                  auto global_idx = std::distance(ms2.data_handle(), &i);
                  dim3(global_idx, ms3);
                  // auto [i1, i2, i3] = dim3(global_idx);
                  ms3(ii, ij, ik) = 1000 + global_idx;
                });

  // read subset of data using 3D view
  for (size_t i = 0; i < ms3.extent(0); i++) {
    for (size_t j = 0; j < 10; j++) {
      for (size_t k = 0; k < ms3.extent(2); k++) {
        assert(ms3(i, j, k) == 1000 + i * ms3.extent(1) * ms3.extent(2) +
                                   j * ms3.extent(2) + k);
        std::cout << ms3(i, j, k) << " ";
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;
  }

  std::cout << ms3(0, 0, 1) << "\n";
}