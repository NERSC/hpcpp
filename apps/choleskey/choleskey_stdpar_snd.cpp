// Cholesky Decomposition: stdpar-->sender
#include <algorithm>
#include <exec/any_sender_of.hpp>
#include <experimental/mdspan>
#include <iostream>
#include <numeric>
#include <stdexec/execution.hpp>
#include <vector>
#include "argparse/argparse.hpp"
#include "commons.hpp"
#include "exec/static_thread_pool.hpp"

#include "matrixutil.hpp"
// using namespace stdexec;

using namespace std;

struct solver {

  using view_2d = std::extents<int, std::dynamic_extent, std::dynamic_extent>;

  template <typename T>
  std::vector<std::vector<T>> Cholesky_Decomposition(std::vector<T>& vec,
                                                     int n) {

    // test here first, scheduler from a thread pool
    exec::static_thread_pool pool(n);
    stdexec::scheduler auto sch = pool.get_scheduler();
    stdexec::sender auto begin = stdexec::schedule(sch);

    std::vector<std::vector<T>> lower(n, std::vector<T>(n, 0));

    auto matrix_ms =
        std::mdspan<T, view_2d, std::layout_right>(vec.data(), n, n);

    auto multiplier_lambda = [=](auto a, auto b) {
      return a * b;
    };

    int np = 3;  // default number of parallel sec, will be an option

    for (int i = 0; i < matrix_ms.extent(0); i++) {
      for (int j = 0; j <= i; j++) {
        T sum = 0;

        if (j == i)  // summation for diagonals
        {
          std::vector<T> sum_vec(np + 1);

          std::cout << "j = " << j << std::endl;

          std::size_t const size = ((j + 1) + (np - 1)) / np;  // partition size
          stdexec::sender auto send1 =
              stdexec::bulk(
                  begin, np,
                  [&](int piece) {
                    std::cout << "pcs = " << piece << std::endl;
                    int start = piece * size;
                    int end = std::min(j, (int)((piece + 1) * size));

                    sum_vec[piece] = std::transform_reduce(
                        std::execution::par, counting_iterator(lower[j][start]),
                        counting_iterator(lower[j][end]), 0, std ::plus{},
                        [=](int val) { return val * val; });
                  }) |
              stdexec::then([&sum_vec]() {
                return std::reduce(std::execution::par, sum_vec.begin(),
                                   sum_vec.end());
              });

          auto [sum1] = stdexec::sync_wait(std::move(send1)).value();

          lower[j][j] = std::sqrt(matrix_ms(i, j) - sum1);

        } else {  // Evaluating L(i, j) using L(j, j)
          // TODO
          sum = std::transform_reduce(std::execution::par, lower[j].cbegin(),
                                      lower[j].cbegin() + j, lower[i].cbegin(),
                                      0, std::plus<>(), multiplier_lambda);

          lower[i][j] = (matrix_ms(i, j) - sum) / lower[j][j];
        }
      }
    }

    return lower;
  }
};

///////////////////////////////////////////////////////////////////////////////
int benchmark(args_params_t const& args) {

  std::uint64_t nd = args.nd;  // Number of matrix dimension.
  std::uint64_t np = args.np;  // Number of partitions.

  std::vector<int> inputMatrix = generate_pascal_matrix<int>(nd);

  // Create the solver object
  solver solve;

  exec::static_thread_pool pool(np);
  stdexec::scheduler auto sch = pool.get_scheduler();
  stdexec::sender auto begin = stdexec::schedule(sch);

  // Measure execution time.
  Timer timer;

  // start decomposation
  auto res_matrix = solve.Cholesky_Decomposition(inputMatrix, nd);

  // Print the final results
  if (args.results) {
    // Displaying Lower Triangular and its Transpose
    cout << setw(6) << " Lower Triangular" << setw(30) << "Transpose" << endl;
    for (int i = 0; i < nd; i++) {
      // Lower Triangular
      for (int j = 0; j < nd; j++)
        cout << setw(6) << res_matrix[i][j] << "\t";
      cout << "\t";

      // Transpose of Lower Triangular
      for (int j = 0; j < nd; j++)
        cout << setw(6) << res_matrix[j][i] << "\t";
      cout << endl;
    }
  }

  if (args.time) {
    std::cout << "Duration: " << time << " ms."
              << "\n";
  }

  return 0;
}

// Driver Code for testing
int main(int argc, char* argv[]) {

  // parse params
  args_params_t args = argparse::parse<args_params_t>(argc, argv);
  // see if help wanted
  if (args.help) {
    args.print();  // prints all variables
    return 0;
  }

  benchmark(args);

  return 0;
}
