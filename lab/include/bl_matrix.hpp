/*!
 * \file bl_matrix.hpp
 * \author Juan Ramirez (juan.ramirez@fing.edu.uy)
 */
#ifndef BL_MATRIX_HPP__
#define BL_MATRIX_HPP__

#include <vector>
#include <iomanip>
#include <cstdint>
#include <cuda_device_runtime_api.h>

template<typename T, size_t Block_Width>
class BLMatrix {
public:
    static const uint32_t block_width = Block_Width;

    const uint32_t rows;
    const uint32_t columns;

    // non-null values, sorted by block-row, then block, then row
    std::vector<value_type> values;

    // index of the start of each block in `values`
    // length: number of non-null blocks + 1
    std::vector<uint32_t> bl_starts;

    // each bit indicates whether the corresponding cell has
    // a non-zero value or not
    // length: number of non-null blocks
    std::vector<uint64_t> bl_bitmaps;

    // block-index in the block-matrix
    // note: the block-matrix has rows/block_width rows and columns/block_width columns
    // length: number of non-null blocks
    std::vector<uint32_t> bl_col_indices;

    // index of the start of each block-row in `bl_starts`, `bl_bitmaps` and `bl_col_indices`
    // length: number block-rows + 1
    std::vector<uint32_t> bl_row_pointers;

    struct DeviceStruct {
        uint32_t rows;
        uint32_t columns;
        T *values;
        uint32_t *bl_starts;
        uint64_t *bl_bitmaps;
        uint32_t *bl_col_indices;
        uint32_t *bl_row_pointers;
    };

    BLMatrix(uint32_t r, uint32_t c)
     : rows(r)
     , columns(c)
    {
        // make sure rows and columns are divisible by the block size
        assert((rows & (block_width-1)) == 0);
        assert((columns & (block_width-1)) == 0);
    }

    BLMatrix(uint32_t r,
             uint32_t c,
             std::vector<value_type> vals,
             std::vector<uint32_t> starts,
             std::vector<uint64_t> bitmaps,
             std::vector<uint32_t> col_indices,
             std::vector<uint32_t> row_pointers)
     : rows(r)
     , columns(c)
     , values(std::move(vals))
     , bl_starts(std::move(starts))
     , bl_bitmaps(std::move(bitmaps))
     , bl_col_indices(std::move(col_indices))
     , bl_row_pointers(std::move(row_pointers))
    {
        // make sure rows and columns are divisible by the block size
        assert((rows & (block_width-1)) == 0);
        assert((columns & (block_width-1)) == 0);
        assert(bl_starts.size() == bl_bitmaps.size() + 1);
        assert(bl_col_indices.size() == bl_bitmaps.size());
        assert(bl_row_pointers.size() == bl_bitmaps.size() / block_width + 1);
    }

    value_type get(uint32_t row, uint32_t column) const {
        assert(row < rows);
        assert(column < columns);

        auto block_y = row / block_width;
        auto block_x = column / block_width;
        auto block_i = row % block_width;
        auto block_j = column % block_width;

        auto col_it_begin = bl_col_indices.begin() + bl_row_pointers[block_y];
        auto col_it_end = bl_col_indices.begin() + bl_row_pointers[block_y+1];
        auto it = std::lower_bound(col_it_begin, col_it_end, block_x);

        if (it != col_it_end && *it == block_x) {
            auto block_idx = it - bl_col_indices.begin();
            auto bitmap = bl_bitmaps[block_idx];
            auto bit = 1ULL << ((block_width*block_width) - (block_i*block_width + block_j) - 1);
            if (bitmap & bit) {
                auto offset = __builtin_popcountll(bitmap - (bitmap&(bit+bit-1)));
                return values[bl_starts[block_idx] + offset];
            }
        }

        return 0;
    }

    void random_init(float non_zero_prob = 0.01, value_type max = 100) {
        // of course this is an approximation
        auto non_null = static_cast<size_t>(non_zero_prob*rows*columns);
        values.reserve(non_null);
        bl_starts.reserve((non_null >> 6) + 1);
        bl_bitmaps.reserve(non_null >> 6);
        bl_col_indices.reserve(non_null >> 6);
        bl_row_pointers.reserve(rows/block_width + 1);

        bl_row_pointers.push_back(0);
        for (uint32_t block_y = 0; block_y < rows; block_y += block_width) {
            for (uint32_t block_x = 0; block_x < columns; block_x += block_width) {
                uint64_t bitmap = 0;

                // if the block is not null, it will start here
                uint32_t bl_start = values.size();

                // fill the block
                for (uint32_t i = 0; i < block_width; ++i) {
                    for (uint32_t j = 0; j < block_width; ++j) {
                        if (rand_unif() < non_zero_prob) {
                            values.push_back((value_type)(int)(rand_unif() * max));
                            bitmap |= (1ULL << (i*block_width + j));
                        }
                    }
                }

                // if the block is not null...
                if (bitmap != 0) {
                    bl_starts.push_back(bl_start);
                    bl_bitmaps.push_back(bitmap);
                    bl_col_indices.push_back(block_x / block_width);
                    bl_row_pointers.push_back(bl_starts.size());
                }
            }
        }
    }

    DeviceStruct to_device() {
        DeviceStruct dMat;
        dMat.rows = rows;
        dMat.columns = columns;
#if 0
        size_t vsize = size_in_bytes(values);
        size_t ssize = size_in_bytes(bl_starts);
        size_t bsize = size_in_bytes(bl_bitmaps);
        size_t csize = size_in_bytes(bl_col_indices);
        size_t rsize = size_in_bytes(bl_row_pointers);

        CUDA_CHK(cudaMalloc(&dMat.values, vsize+ssize+bsize+csize+rsize));
        CUDA_CHK(cudaMemcpy(dMat.values, values.data(), vsize, cudaMemcpyHostToDevice));
        dMat.bl_starts = reinterpret_cast<uint32_t *>(dMat.values + vsize);
        CUDA_CHK(cudaMemcpy(dMat.bl_starts, bl_starts.data(), ssize, cudaMemcpyHostToDevice));
        dMat.bl_bitmaps = reinterpret_cast<uint64_t *>(dMat.bl_starts + ssize);
        CUDA_CHK(cudaMemcpy(dMat.bl_bitmaps, bl_bitmaps.data(), bsize, cudaMemcpyHostToDevice));
        dMat.bl_col_indices = reinterpret_cast<uint32_t *>(dMat.bl_bitmaps + bsize);
        CUDA_CHK(cudaMemcpy(dMat.bl_col_indices, bl_col_indices.data(), csize, cudaMemcpyHostToDevice));
        dMat.bl_row_pointers = reinterpret_cast<uint32_t *>(dMat.bl_col_indices + csize);
        CUDA_CHK(cudaMemcpy(dMat.bl_row_pointers, bl_row_pointers.data(), rsize, cudaMemcpyHostToDevice));

        return dMat;
#else
        CUDA_CHK(cudaMalloc(&dMat.values, size_in_bytes(values)));
        CUDA_CHK(cudaMemcpy(dMat.values, values.data(), size_in_bytes(values), cudaMemcpyHostToDevice));

        CUDA_CHK(cudaMalloc(&dMat.bl_starts, size_in_bytes(bl_starts)));
        CUDA_CHK(cudaMemcpy(dMat.bl_starts, bl_starts.data(), size_in_bytes(bl_starts), cudaMemcpyHostToDevice));

        CUDA_CHK(cudaMalloc(&dMat.bl_bitmaps, size_in_bytes(bl_bitmaps)));
        CUDA_CHK(cudaMemcpy(dMat.bl_bitmaps, bl_bitmaps.data(), size_in_bytes(bl_bitmaps), cudaMemcpyHostToDevice));

        CUDA_CHK(cudaMalloc(&dMat.bl_col_indices, size_in_bytes(bl_col_indices)));
        CUDA_CHK(cudaMemcpy(dMat.bl_col_indices, bl_col_indices.data(), size_in_bytes(bl_col_indices), cudaMemcpyHostToDevice));

        CUDA_CHK(cudaMalloc(&dMat.bl_row_pointers, size_in_bytes(bl_row_pointers)));
        CUDA_CHK(cudaMemcpy(dMat.bl_row_pointers, bl_row_pointers.data(), size_in_bytes(bl_row_pointers), cudaMemcpyHostToDevice));

        return dMat;
#endif
    }

    void device_free(DeviceStruct dMat) {
        cudaFree(dMat.values);
#if 0
        cudaFree(dMat.bl_starts);
        cudaFree(dMat.bl_bitmaps);
        cudaFree(dMat.bl_col_indices);
        cudaFree(dMat.bl_row_pointers);
#endif
    }
};

// ostream operator for BLMatrix
template<typename T, size_t Block_Width>
std::ostream& operator<<(std::ostream& os, const BLMatrix<T, Block_Width>& m) {
    auto w = os.width();
    for (uint32_t i = 0; i < m.rows; ++i) {
        for (uint32_t j = 0; j < m.columns; ++j) {
            os << std::setw(3) << m.get(i, j) << " ";
            m.get(i, j);
        }
        if (i < m.rows-1) {
            os << std::endl;
        }
    }
    os.width(w);
    return os;
}

template<class T> using BLMatrix2 = BLMatrix<T, 2>;

template<class T> using BLMatrix8 = BLMatrix<T, 8>;

#endif // BL_MATRIX_HPP__
