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

/*
 * A simplified version of the 3d heat equation example.
 */

#include "commons.hpp"
#include "argparse/argparse.hpp"
#include <experimental/mdspan>


constexpr int dims = 2;
using Real_t = double;

// 2D view
using 2d_view = std::extents<int, std::dynamic_extent, std::dynamic_extent>;

// 3D view
using 3d_view = std::extents<int, std::dynamic_extent, std::dynamic_extent, std::dynamic_extent>;

// parameters
struct heat_params_t : public argparse::Args
{
    int &n_cell = kwarg("n,ncells", "number of cells on each side of the domain").set_default(32);
    int &nsteps = kwarg("s,nsteps", "total steps in simulation").set_default(100);
    Real_t &alpha = kwarg("a,alpha", "thermal diffusivity").set_default(0.5f);
    Real_t &dt = kwarg("t,dt", "time step").set_default(1.0e-5f);
    bool &help = kwarg("h, help", "print help").set_default(false);
    // future use if needed
    //int &max_grid_size = kwarg("g, max_grid_size", "size of each box (or grid)").set_default(32);
    // bool &verbose = kwarg("v, verbose", "verbose mode").set_default(false);
    // int &plot_int = kwarg("p, plot_int", "how often to write a plotfile").set_default(-1);
};


int main(int argc, char *argv[])
{
    // parse params
    heat_params_t args = argparse::parse<heat_params_t>(argc, argv);

    // see if help wanted
    if (args.help)
    {
        args.print(); // prints all variables
        return 0;
    }

    // simulation variables
    int n_cell = args.n_cell;
    int nsteps = args.nsteps;
    Real_t dt = args.dt;
    Real_t alpha = args.alpha;
    // future if needed to split in multiple grids
    // int max_grid_size = args.max_grid_size;

    // number of dimensions
    constexpr int dims = 2;
    // total number of ghost cells = ghosts x dims
    constexpr int nghosts = dims * 1;

    const int grid_size = n_cell * n_cell;

    // initialize dx, dy, dz
    auto *dx = new Real_t[dims];
    for (int i = 0; i < dims; ++i)
        dx[0] = 1.0 / (n_cell - 1);

    // simulation setup (2D)
    Real_t *old_grid = new Real_t[(n_cell+nghosts) * (n_cell+nghosts)];
    Real_t *grid_new = new Real_t[(n_cell+nghosts) * (n_cell+nghosts)];

    auto phi_old = std::mdspan<int, 2d_view, std::layout_right> (old_grid, ncells + nghosts, ncells + nghosts);
    auto phi_new = std::mdspan<int, 2d_view, std::layout_right> (grid_new, ncells, ncells);

    // init simulation time
    Real_t time = 0.0;


    // delete all memory
    delete[] indices;
    delete[] phi_old;
    delete[] phi_new;

    indices = nullptr;
    phi_old = nullptr;
    phi_new = nullptr;

    return 0;
}
