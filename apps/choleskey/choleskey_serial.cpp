// Cholesky Decomposition:  mdspan
#include <bits/stdc++.h>
#include <experimental/mdspan>
#include <vector>
#include "argparse/argparse.hpp"
#include "commons.hpp"
#include "matrixutil.hpp"

using namespace std;

struct solver {

  using view_2d = std::extents<int, std::dynamic_extent, std::dynamic_extent>;

  typedef std::mdspan<int, view_2d, std::layout_right> matrix_ms_t;

  template <typename T>
  matrix_ms_t Cholesky_Decomposition(std::vector<T>& vec, int n) {
    std::vector<T> lower(n * n, 0);

    auto matrix_ms =
        std::mdspan<T, view_2d, std::layout_right>(vec.data(), n, n);
    auto lower_ms =
        std::mdspan<T, view_2d, std::layout_right>(lower.data(), n, n);

    // Decomposing a matrix into Lower Triangular
    for (int i = 0; i < matrix_ms.extent(0); i++) {
      for (int j = 0; j <= i; j++) {
        T sum = 0;

        if (j == i) {
          // summation for diagonals
          for (int k = 0; k < j; k++)
            sum += pow(lower_ms(j, k), 2);
          lower_ms(j, j) = sqrt(matrix_ms(i, j) - sum);
        } else {
          // Evaluating L(i, j) using L(j, j)
          for (int k = 0; k < j; k++)
            sum += (lower_ms(i, k) * lower_ms(j, k));
          lower_ms(i, j) = (matrix_ms(i, j) - sum) / lower_ms(j, j);
        }
      }
    }
    return lower_ms;
  }
};

///////////////////////////////////////////////////////////////////////////////
int benchmark(args_params_t const& args) {

  std::uint64_t nd = args.nd;  // Number of matrix dimension.

  std::vector<int> inputMatrix = generate_pascal_matrix<int>(nd);

  // Create the solverobject
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
        cout << setw(6) << res_matrix(i, j) << "\t";
      cout << "\t";

      // Transpose of Lower Triangular
      for (int j = 0; j < nd; j++)
        cout << setw(6) << res_matrix(j, i) << "\t";
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
