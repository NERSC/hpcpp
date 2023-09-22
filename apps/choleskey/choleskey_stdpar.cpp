// Cholesky Decomposition: stdpar
#include "argparse/argparse.hpp"
#include "commons.hpp"

#include <algorithm>
#include <execution>
#include <experimental/mdspan>
#include <iostream>
#include <numeric>
#include <vector>

#include "matrixutil.hpp"

using namespace std;

struct solver {

  using view_2d = std::extents<int, std::dynamic_extent, std::dynamic_extent>;

  template <typename T>
  std::vector<std::vector<T>> Cholesky_Decomposition(std::vector<T>& vec,
                                                     int n) {
    std::vector<std::vector<T>> lower(n, std::vector<T>(n, 0));

    auto matrix_ms =
        std::mdspan<T, view_2d, std::layout_right>(vec.data(), n, n);

    auto multiplier_lambda = [=](auto a, auto b) {
      return a * b;
    };

    // Decomposing a matrix into Lower Triangular
    for (int i = 0; i < matrix_ms.extent(0); i++) {
      for (int j = 0; j <= i; j++) {
        T sum = 0;

        if (j == i)  // summation for diagonals
        {
          sum = std::transform_reduce(std::execution::par, lower[j].cbegin(),
                                      lower[j].cbegin() + j, 0, std::plus{},
                                      [=](int val) { return val * val; });

          lower[j][j] = std::sqrt(matrix_ms(i, j) - sum);

        } else {  // Evaluating L(i, j) using L(j, j)

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

  std::vector<int> inputMatrix = generate_pascal_matrix<int>(nd);

  // Create the solver object
  solver solve;
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
