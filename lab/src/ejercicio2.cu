/*!
 * \file ejercicio2.cu
 * \author Juan Ramirez (juan.ramirez@fing.edu.uy)
 */

#include "lab.hpp"
#include "bl_matrix.hpp"


template <typename value_type, size_t Block_Width>
__host__ __device__ void build_dense_block(uint32_t block_column,
                                           const uint64_t * __restrict__ bl_bitmaps,
                                           const uint32_t * __restrict__ bl_starts,
                                           const value_type * __restrict__ values,
                                           value_type *block)
{
    auto bitmap = bl_bitmaps[block_column];
    auto start = bl_starts[block_column];
    auto offset = 0u;
    for (auto i = 0u; i < Block_Width*Block_Width; ++i) {
        auto bit = 1ULL << ((Block_Width*Block_Width) - i - 1);
        if (bitmap & bit) {
            block[i] = values[start + offset];
            ++offset;
        }
    }
}

// Note: Same as standard matrix multiplication
// Will work better only if there are several empty blocks
template <typename value_type, size_t Block_Width>
__host__ void serial_spmv_kernel_host(const BLMatrix<value_type, Block_Width>& mat,
                                      const value_type * __restrict__ x,
                                      value_type *y)
{
    for (auto i = 0u; i < mat.rows / Block_Width; ++i) {
        auto block_row = i * Block_Width;
        // j runs though the block columns in the current row
        for (auto j = mat.bl_row_pointers[i]; j < mat.bl_row_pointers[i + 1]; ++j) {
            auto block_col = mat.bl_col_indices[j] * Block_Width;

            value_type block[Block_Width * Block_Width] = {0};
            build_dense_block<value_type, Block_Width>(j,
                                                       mat.bl_bitmaps.data(),
                                                       mat.bl_starts.data(),
                                                       mat.values.data(),
                                                       block);
            for (auto k = 0u; k < Block_Width; ++k) {
                for (auto l = 0u; l < Block_Width; ++l) {
                    y[block_row + k] += block[k * Block_Width + l] * x[block_col + l];
                }
            }
        }
    }
}

/////////////////////////////////
         ///  IDEA  ///
/////////////////////////////////
// Que bloque denso sea un CSR //
/////////////////////////////////


static void initial_kindergarten_test() {
    constexpr uint32_t rows = 6;
    constexpr uint32_t cols = 6;

    std::vector<value_type> values{{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}};
    std::vector<uint32_t> bl_starts{{0, 2, 3, 4, 5, 7, 8, 12}};
    std::vector<uint64_t> bl_bitmaps{{9, 2, 2, 1, 3, 1, 15}};
    std::vector<uint32_t> bl_col_indices{{0, 1, 2, 0, 2, 0, 1}};
    std::vector<uint32_t> bl_row_pointers{{0, 3, 5, 7}};

    std::vector<value_type> x{{1, 1, 2, 1, 1, 2}};
    std::vector<value_type> y{{0, 0, 0, 0, 0, 0}};
    std::vector<value_type> y_expected{{1, 12, 0, 25, 28, 42}};

    BLMatrix2<value_type> mat{rows,
                              cols,
                              values,
                              bl_starts,
                              bl_bitmaps,
                              bl_col_indices,
                              bl_row_pointers};

    std::cout << "Matrix:" << std::endl;
    std::cout << mat << std::endl;

    serial_spmv_kernel_host<value_type, 2>(mat, x.data(), &y[0]);

    // print y
    std::cout << "y: ";
    for (auto i = 0u; i < y.size(); ++i) {
        std::cout << y[i] << " ";
    }
    std::cout << std::endl;
}

void ejercicio2() {
    initial_kindergarten_test();
}
