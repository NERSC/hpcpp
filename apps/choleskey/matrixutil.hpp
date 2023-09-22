#pragma once

#include <iostream>
#include <vector>

// generate positive definition matrix
template <typename T>
using Matrix = std::vector<std::vector<T>>;

template <typename T>
std::vector<T> generate_pascal_matrix(const int n) {
  Matrix<T> matrix(n, std::vector<T>(n, static_cast<T>(0)));

  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      if (i == 0 || j == 0) {
        matrix[i][j] = static_cast<T>(1);
      } else {
        matrix[i][j] = matrix[i][j - 1] + matrix[i - 1][j];
      }
    }
  }

  std::vector<T> flattenedVector;
  for (const auto& row : matrix) {
    flattenedVector.insert(flattenedVector.end(), row.begin(), row.end());
  }
  return std::move(flattenedVector);
}

// parameters define
struct args_params_t : public argparse::Args {
  bool& results = kwarg("results", "print generated results (default: false)")
                      .set_default(true);
  std::uint64_t& nd =
      kwarg("nd", "Number of input(positive definition) matrix dimension(<=18)")
          .set_default(10);

  bool& help = flag("h, help", "print help");
  bool& time = kwarg("t, time", "print time").set_default(true);
};
