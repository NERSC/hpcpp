#include <span>
#include <vector>
#include <experimental/mdspan>
#include <iostream>

using data_type    = int;
// 2D view
using extents_type = std::extents<data_type, std::dynamic_extent, std::dynamic_extent>;
// 3D view
using extents_type2 = std::extents<data_type, std::dynamic_extent, std::dynamic_extent, std::dynamic_extent>;

int main()
{
    std::vector<int> v = {1,2,3,4,5,6,7,8,9,10,11,12};

    // View data as contiguous memory representing 2 rows of 6 ints each
    auto ms2 = std::mdspan<data_type, extents_type, std::layout_left> (v.data(), 2, 6);
    // View the same data as a 3D array 2 x 3 x 2
    auto ms3 = std::mdspan<data_type, extents_type2, std::layout_left> (v.data(), 2, 3, 2);

    // write data using 2D view
    for(size_t i=0; i != ms2.extent(0); i++)
    {
        for(size_t j=0; j != ms2.extent(1); j++)
        {
            ms2(i, j) = 1000 + ms2(i, j);
            std::cout << ms2(i, j) << " ";
        }
        std::cout << std::endl;
    }

    std::cout << std::endl << std::endl;

    // write data using 3D view
    for(size_t i=0; i<ms3.extent(0); i++){
        for(size_t j=0; j<ms3.extent(1); j++)
        {
            for(size_t k=0; k<ms3.extent(2); k++){
                ms3(i, j, k) = 1000 + ms3(i, j, k);
                std::cout << ms3(i, j, k) << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }


    std::cout << "\n";
}