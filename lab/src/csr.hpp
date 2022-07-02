/*!
 * \file csr.hpp
 * \author Juan Ramirez (juan.ramirez@fing.edu.uy)
 */
#ifndef CSR_HPP__
#define CSR_HPP__

#include <vector>
#include <array>

#include "utils.h"

template <typename Value_Type>
class CSRMatrix {
public:
    typedef Value_Type value_type;
    uint32_t rows;
    uint32_t columns;
    std::vector<value_type> values;
    std::vector<uint32_t> col_indices;
    std::vector<uint32_t> row_pointers;

    struct DeviceCSRMatrix {
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

    void random_init(float non_zero_prob = 0.05,  value_type max = 100) {
        values.reserve(static_cast<size_t>(non_zero_prob*rows*columns));
        col_indices.reserve(static_cast<size_t>(non_zero_prob*rows*columns));
        row_pointers.reserve(rows+1);

        uint32_t counter = 0;

        row_pointers.push_back(0);
        for (uint32_t i = 0; i < rows; ++i) {
            for (uint32_t j = 0; j < columns; ++j) {
                if (genrand() <= non_zero_prob) {
                    values.push_back(static_cast<value_type>(genrand() * max));
                    col_indices.push_back(j);
                    ++counter;
                }
            }
            row_pointers.push_back(counter);
        }
    }

    DeviceCSRMatrix to_device() {
        DeviceCSRMatrix dMat;
        dMat.rows = rows;
        cudaMalloc(&dMat.values, values.size()*sizeof(value_type));
        cudaMalloc(&dMat.col_indices, col_indices.size()*sizeof(uint32_t));
        cudaMalloc(&dMat.row_pointers, row_pointers.size()*sizeof(uint32_t));

        cudaMemcpy(dMat.values, values.data(), values.size()*sizeof(value_type), cudaMemcpyHostToDevice);
        cudaMemcpy(dMat.col_indices, col_indices.data(), col_indices.size()*sizeof(uint32_t), cudaMemcpyHostToDevice);
        cudaMemcpy(dMat.row_pointers, row_pointers.data(), row_pointers.size()*sizeof(uint32_t), cudaMemcpyHostToDevice);
        return dMat;
    }

    void device_free(DeviceCSRMatrix dMat) {
        cudaFree(dMat.values);
        cudaFree(dMat.col_indices);
        cudaFree(dMat.row_pointers);
    }
};

#endif // CSR_HPP__
