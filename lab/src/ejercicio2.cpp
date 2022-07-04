/*!
 * \file ejercicio2.cuh
 * \author Juan Ramirez (juan.ramirez@fing.edu.uy)
 */
#ifndef EJERCICIO2_CUH__
#define EJERCICIO2_CUH__

#include "lab.hpp"
#include "bl_matrix.hpp"

void bl_initial_algorithm_test() {
    //constexpr size_t BL_SIZE = 2;
    constexpr uint32_t rows = 6;
    constexpr uint32_t cols = 6;
    //constexpr uint32_t bl_rows = rows / BL_SIZE;

    std::vector<value_type> values{{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}};
    std::vector<uint32_t> bl_starts{{0, 2, 3, 4, 5, 7, 8, 12}};
    std::vector<uint64_t> bl_bitmaps{{9, 2, 2, 1, 3, 1, 15}};
    std::vector<uint32_t> bl_col_indices{{0, 1, 2, 0, 2, 0, 1}};
    std::vector<uint32_t> bl_row_pointers{{0, 3, 5, 7}};

    BLMatrix2<value_type> mat{rows,
                              cols,
                              values,
                              bl_starts,
                              bl_bitmaps,
                              bl_col_indices,
                              bl_row_pointers};

    std::cout << "Matrix:" << std::endl;
    std::cout << mat << std::endl;
}

void ejercicio2() {
    bl_initial_algorithm_test();
}

#endif // EJERCICIO2_CUH__
