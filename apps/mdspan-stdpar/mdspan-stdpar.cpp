#include <vector>
#include <span>
#include "mdspan/mdspan.hpp"
#include <fmt/core.h>

int main()
{
    // Class template argument deduction (CTAD) (since C++17)
    std::vector v = {1,2,3,4};

    // View v as contiguous memory representing 4 ints
    auto s = std::span(v.data(), 4);

    // View v as contiguous memory representing 2 rows
    // of 2 ints each.
    // Eventually, we will use std::mdspan provided by the standard
    auto ms = std::experimental::mdspan(v.data(),2,2);

    // access data via []operator
    for(size_t i=0; i<ms.extent(0); i++)
        for(size_t j=0; j<ms.extent(1); j++)
            ms[i, j] = 1000 + ms[i, j];
}
