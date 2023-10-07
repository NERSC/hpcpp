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

#include "commons.hpp"
#include "exec/static_thread_pool.hpp"

using namespace std;
using namespace stdexec;
using stdexec::sync_wait;

using T = double;
using time_point_t = std::chrono::system_clock::time_point;

// must take in the pointers/vectors by reference
template <typename P>
auto work(P& A, P& B, P& Y, int N) {
  T sum = 0.0;

  // init A and B separately - will it cause an H2D copy?
  sender auto s1 = then(just(),
                        [&] {
                          std::for_each(std::execution::par_unseq, &A[0], &A[N],
                                        [&](T& ai) { ai = cos(M_PI / 4); });
                        })
                   // trigger a D2H here
                   | then([&] {
                       for (int i = 0; i < N / 3; i++) {
                         // read only or read-write operations
                         sum += A[i] / N;

                         // this line if commented should not result in an H2D
                         // after this but it does.
                         // A[i] = sin(M_PI/4);
                       }
                       std::cout << std::endl;
                     });

  // will it cause an H2D here?
  sender auto s2 = then(just(), [&] {
    std::for_each(std::execution::par_unseq, &B[0], &B[N],
                  [&](T& bi) { bi = sin(M_PI / 6); });
  });

  // will s1 and s2 execute in parallel or not?
  sync_wait(when_all(std::move(s1), std::move(s2)));

  // compute Y = sqrt((A+B)^2 + B^2)/(A+B+B)
  sender auto s3 =
      then(just(),
           [&] {
             std::transform(std::execution::par_unseq, &A[0], &A[N], &B[0],
                            &A[0], [&](T& ai, T& bi) { return ai + bi; });
             std::transform(std::execution::par_unseq, &A[0], &A[N], &B[0],
                            &Y[0], [&](T& ai, T& bi) {
                              return sqrt(pow(ai, 2) + pow(bi, 2)) / (ai + bi);
                            });
           })
      // should trigger a D2H copy of N/3 elements
      | then([&] {
          for (int i = 0; i < N / 3; i++)
            sum += Y[i] / N;

          std::cout << std::endl;
        })
      // get sum(Y) - wonder if there is another H2D as we only read it in the
      // last step
      | then([&] {
          return std::reduce(std::execution::par_unseq, &Y[0], &Y[N], 0.0,
                             std::plus<T>());
        });

  auto [val] = sync_wait(s3).value();

  return sum += val;
}

int main(int argc, char* argv[]) {
  constexpr int N = 1e9;
  time_point_t mark = std::chrono::system_clock::now();
  auto es =
      std::chrono::duration<double>(std::chrono::system_clock::now() - mark)
          .count();
  T sum = 0.0;

#if 1  // 0 if only arrays
  std::vector<T> A(N);
  std::vector<T> B(N);
  std::vector<T> Y(N);

  mark = std::chrono::system_clock::now();
  sum = work(A, B, Y, N);
  es = std::chrono::duration<double>(std::chrono::system_clock::now() - mark)
           .count();
  std::cout << "Vectors: Elapsed Time: " << es << "s" << std::endl << std::endl;

  std::cout << fixed << "sum: " << sum << "\n";
#endif

#if 1  // 0 if only vectors

  // allocate memory - can we just allocate it on device only?
  T* a = new T[N];
  T* b = new T[N];
  T* y = new T[N];

  sum = 0;
  mark = std::chrono::system_clock::now();
  sum = work(a, b, y, N);
  es = std::chrono::duration<double>(std::chrono::system_clock::now() - mark)
           .count();
  std::cout << "Pointers: Elapsed Time: " << es << "s" << std::endl
            << std::endl;

  // do not use scientific notation
  std::cout << fixed << "sum: " << sum << "\n";
#endif

  return 0;
}