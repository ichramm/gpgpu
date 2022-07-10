/*!
 * \file csr_matrix.hpp
 * \author Juan Ramirez (juan.ramirez@fing.edu.uy)
 */
#ifndef CSR_MATRIX_HPP__
#define CSR_MATRIX_HPP__

#include <vector>
#include <iomanip>
#include <cuda_device_runtime_api.h>

template<typename T>
class CSRMatrix {
public:
    typedef T value_type;

    uint32_t rows;
    uint32_t columns;
    std::vector<value_type> values;
    std::vector<uint32_t> col_indices;
    std::vector<uint32_t> row_pointers;

    struct DeviceStruct {
        uint32_t rows;
        value_type *values;
        uint32_t *col_indices;
        uint32_t *row_pointers;
    };

    CSRMatrix(uint32_t r, uint32_t c)
     : rows(r)
     , columns(c)
    {}

    CSRMatrix(uint32_t r,
              uint32_t c,
              std::vector<value_type> vals,
              std::vector<uint32_t> colIdx,
              std::vector<uint32_t> rowPtr)
     : rows(r),
       columns(c),
       values(std::move(vals)),
       col_indices(std::move(colIdx)),
       row_pointers(std::move(rowPtr))
    {
        assert(values.size() == col_indices.size());
        assert(row_pointers.size() == rows+1);
    }

    value_type get(uint32_t row, uint32_t column) const {
        assert(row < rows);
        assert(column < columns);
        uint32_t start = row_pointers[row];
        uint32_t end = row_pointers[row+1];
        for (uint32_t i = start; i < end; i++) {
            if (col_indices[i] == column) {
                return values[i];
            }
        }
        return 0;
    }

    void random_init(float non_zero_prob = 0.01, value_type max = 100) {
        // of course this is an approximation
        auto non_null = static_cast<size_t>(non_zero_prob*rows*columns);
        values.reserve(static_cast<size_t>(non_null));
        col_indices.reserve(static_cast<size_t>(non_null));
        row_pointers.reserve(rows+1);

        uint32_t counter = 0;

        row_pointers.push_back(0);
        for (uint32_t i = 0; i < rows; ++i) {
            for (uint32_t j = 0; j < columns; ++j) {
                if (rand_unif() < non_zero_prob) {
                    values.push_back(static_cast<value_type>(rand_unif() * max));
                    col_indices.push_back(j);
                    ++counter;
                }
            }
            row_pointers.push_back(counter);
        }
    }

    DeviceStruct to_device() {
        DeviceStruct dMat;
        dMat.rows = rows;
#if 0
        dMat.values = dev_alloc_fill(values.size(), values.data()).release();
        dMat.col_indices = dev_alloc_fill(col_indices.size(), col_indices.data()).release();
        dMat.row_pointers = dev_alloc_fill(row_pointers.size(), row_pointers.data()).release();
#else
        CUDA_CHK(cudaMalloc(&dMat.values, size_in_bytes(values)));
        CUDA_CHK(cudaMalloc(&dMat.col_indices, size_in_bytes(col_indices)));
        CUDA_CHK(cudaMalloc(&dMat.row_pointers, size_in_bytes(row_pointers)));

        CUDA_CHK(cudaMemcpy(dMat.values, values.data(), size_in_bytes(values), cudaMemcpyHostToDevice));
        CUDA_CHK(cudaMemcpy(dMat.col_indices, col_indices.data(), size_in_bytes(col_indices), cudaMemcpyHostToDevice));
        CUDA_CHK(cudaMemcpy(dMat.row_pointers, row_pointers.data(), size_in_bytes(row_pointers), cudaMemcpyHostToDevice));
#endif
        return dMat;
    }

    void device_free(DeviceStruct dMat) {
        CUDA_CHK(cudaFree(dMat.values));
        CUDA_CHK(cudaFree(dMat.col_indices));
        CUDA_CHK(cudaFree(dMat.row_pointers));
    }
};

template<typename T>
std::ostream& operator<<(std::ostream& os, const CSRMatrix<T>& m) {
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

#endif // CSR_MATRIX_HPP__
