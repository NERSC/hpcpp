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
 * commons for the heat equation examples
 */

#pragma once

#include <experimental/mdspan>

#include "argparse/argparse.hpp"
#include "commons.hpp"

// data type
using Real_t = double;

// number of dimensions
constexpr int dims = 2;

// total number of ghost cells = ghosts x dims
constexpr int ghost_cells = 1;
constexpr int nghosts = ghost_cells * dims;

// 2D view
using view_2d = std::extents<int, std::dynamic_extent, std::dynamic_extent>;

// 3D view
using view_3d = std::extents<int, std::dynamic_extent, std::dynamic_extent,
                             std::dynamic_extent>;

// macros to get x and y positions from indices
#define pos(i, ghosts, dx) -0.5 + dx*(i - ghosts)

// parameters
struct heat_params_t : public argparse::Args {
  int& ncells = kwarg("n,ncells", "number of cells on each side of the domain")
                    .set_default(32);
  int& nsteps = kwarg("s,nsteps", "total steps in simulation").set_default(100);
  Real_t& alpha = kwarg("a,alpha", "thermal diffusivity").set_default(0.5f);
  Real_t& dt = kwarg("t,dt", "time step").set_default(5.0e-5f);
  bool& help = flag("h, help", "print help");
  bool& print_grid = flag("p,print", "print grids at step 0 and step n");
#if defined(TILING)
  int& ntiles = kwarg("ntiles", "number of parallel tiles").set_default(4);
#endif  // TILING               \
        // future use if needed \
        // int &max_grid_size = kwarg("g, max_grid_size", "size of each box (or
  // grid)").set_default(32); bool &verbose = kwarg("v, verbose", "verbose
  // mode").set_default(false); int &plot_int = kwarg("p, plot_int", "how often
  // to write a plotfile").set_default(-1);
};

template <typename T>
void printGrid(T* grid, int len) {
  auto view = std::mdspan<T, view_2d, std::layout_right>(grid, len, len);
  std::cout << "Grid: " << std::endl;
  std::cout << std::fixed << std::showpoint;
  std::cout << std::setprecision(2);

  for (auto j = 0; j < view.extent(1); ++j) {
    for (auto i = 0; i < view.extent(0); ++i) {
      std::cout << view(i, j) << ", ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
}

// fill boundary cells
template <typename T>
void fill2Dboundaries(T* grid, int len, int ghost_cells = 1) {
  std::for_each_n(std::execution::par_unseq, counting_iterator(ghost_cells),
                  len - nghosts, [=](auto i) {
                    grid[i] = grid[i + (ghost_cells * len)];
                    grid[i + (len * (len - ghost_cells))] =
                        grid[i + (len * (len - ghost_cells - 1))];
                  });

  std::for_each_n(std::execution::par_unseq, counting_iterator(ghost_cells),
                  len - nghosts, [=](auto j) {
                    grid[j * len] = grid[(ghost_cells * len) + j];
                    grid[(len - ghost_cells) + (len * j)] =
                        grid[(len - ghost_cells - 1) + (len * j)];
                  });
}