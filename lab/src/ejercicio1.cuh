/*!
 * \file ejercicio1.cuh
 * \author Juan Ramirez (juan.ramirez@fing.edu.uy)
 */
#ifndef EJERCICIO1_CUH__
#define EJERCICIO1_CUH__

#include "lab.h"
#include "csr.hpp"

#include <iostream>
#include <algorithm>
#include <functional>

// would be faster if it were a host function
template <typename value_type>
__global__ void serial_mult_kernel(typename CSRMatrix<value_type>::DeviceCSRMatrix mat,
                                   value_type *in,
                                   value_type *out) { // already zeroed
    for (uint32_t i = 0; i < mat.rows; ++i) {
        for (uint32_t j = mat.row_pointers[i]; j < mat.row_pointers[i+1]; ++j) {
            out[i] += mat.values[j] * in[mat.col_indices[j]];
        }
    }
}

// on thread per row (no shared mem)
__global__ void par_mult_kernel1(typename CSRMatrix<value_type>::DeviceCSRMatrix mat,
                                 value_type *in,
                                 value_type *out) {
    // 1D grid of 1D blocks
    uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < mat.rows) {
        for (uint32_t j = mat.row_pointers[row]; j < mat.row_pointers[row+1]; ++j) {
            out[row] += mat.values[j] * in[mat.col_indices[j]];
        }
    }
}

// one block per row
__global__ void par_mult_kernel2(typename CSRMatrix<value_type>::DeviceCSRMatrix mat,
                                 value_type *in,
                                 value_type *out) {
    __shared__ value_type row_total;

    if (threadIdx.x == 0) {
        row_total = 0;
    }
    __syncthreads();

    // 1D grid
    uint32_t row = blockIdx.x;

    if (row < mat.rows) {
        value_type acc = 0;
        for (uint32_t j = mat.row_pointers[row]+threadIdx.x; j < mat.row_pointers[row+1]; j += blockDim.x) {
            acc += mat.values[j] * in[mat.col_indices[j]];
        }
        atomicAdd(&row_total, acc);
    }

    __syncthreads();
    if (threadIdx.x == 0) {
        out[row] = row_total;
    }
}

void validate_results(value_type *d_expectedResult,
                      value_type *d_out,
                      uint32_t N) {
    if (gpu_compare_arrays(d_expectedResult, d_out, N)) {
        value_type *h_out = new value_type[N];
        cudaMemcpy(h_out, d_out, sizeof(h_out), cudaMemcpyDeviceToHost);
        for (uint32_t i = 0; i < N; ++i) {
            std::cout << " " << h_out[i];
        }
        std::cout << std::endl;
        delete[] h_out;
    } else {
        std::cout << "OK" << std::endl;
    }
}


void initial_algorithm_test() {
    const uint32_t N = 6;
    const uint32_t C = 5;

    std::vector<value_type> h_vals{{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}};
    std::vector<uint32_t> h_colIdx{{0, 1, 2, 4, 1, 2, 3, 4, 3, 1, 4}};
    std::vector<uint32_t> h_rowPtr{{0, 1, 4, 4, 8, 9, 11}}; // N+1 elements (last is out of vals's bounds)
    value_type h_in[C] = {1, 1, 1, 1, 1};
    value_type h_expectedResult[N] = {1, 9, 0, 26, 9, 21}; // column sum

    value_type *d_in, *d_out, *d_expectedResult;

    CSRMatrix<value_type> mat{6u, 5u, h_vals, h_colIdx, h_rowPtr};

    CUDA_CHK(cudaMalloc(&d_in, sizeof(h_in)));
    CUDA_CHK(cudaMemcpy(d_in, h_in, sizeof(h_in), cudaMemcpyHostToDevice));
    CUDA_CHK(cudaMalloc(&d_out, sizeof(value_type)*N));
    CUDA_CHK(cudaMemset(d_out, 0, sizeof(value_type)*N));

    CUDA_CHK(cudaMalloc(&d_expectedResult, sizeof(h_expectedResult)));
    CUDA_CHK(cudaMemcpy(d_expectedResult, h_expectedResult, sizeof(h_expectedResult), cudaMemcpyHostToDevice));

    auto dMat = mat.to_device();

    cudaMemset(d_out, 0, sizeof(value_type)*N);
    serial_mult_kernel<<<1, 1>>>(dMat, d_in, d_out);
    CUDA_CHK(cudaGetLastError());
    CUDA_CHK(cudaDeviceSynchronize());
    validate_results(d_expectedResult, d_out, N);

    cudaMemset(d_out, 0, sizeof(value_type)*N);
    par_mult_kernel1<<<1, N>>>(dMat, d_in, d_out);
    CUDA_CHK(cudaGetLastError());
    CUDA_CHK(cudaDeviceSynchronize());
    validate_results(d_expectedResult, d_out, N);

    cudaMemset(d_out, 0, sizeof(value_type)*N);
    par_mult_kernel2<<<N, 32>>>(dMat, d_in, d_out);
    CUDA_CHK(cudaGetLastError());
    CUDA_CHK(cudaDeviceSynchronize());
    validate_results(d_expectedResult, d_out, N);

    mat.device_free(dMat);
}

void ejercicio1() {
    initial_algorithm_test();

    const unsigned int ROWS = 10000u;
    const unsigned int COLUMNS = 5000u;

    CSRMatrix<value_type> mat{ROWS, COLUMNS};
    mat.random_init();

    value_type h_in[COLUMNS];
    std::fill(std::begin(h_in), std::end(h_in), 1);

    value_type *d_in, *d_out;

    CUDA_CHK(cudaMalloc(&d_in, sizeof(value_type)*COLUMNS));
    CUDA_CHK(cudaMemcpy(d_in, h_in, sizeof(value_type)*COLUMNS, cudaMemcpyHostToDevice));

    CUDA_CHK(cudaMalloc(&d_out, sizeof(value_type)*ROWS));

    auto dMat = mat.to_device();

    auto test_kernel = [&](std::string name, std::function<void()> f, int times = BENCH_TIMES) {
        Metric m;
        for (auto i = 0; i < times; ++i) {
            CUDA_CHK(cudaMemset(d_out, 0, sizeof(value_type)*ROWS));
            auto t = m.track_begin();
            f();
            CUDA_CHK(cudaGetLastError());
            CUDA_CHK(cudaDeviceSynchronize());
            m.track_end(t);
        }
        printf("%s - mean: %f ms, stdev: %f, CV: %f\n", name.c_str(), m.mean(), m.stdev(), m.cv());
    };

    test_kernel("Serial", [&]() {
        serial_mult_kernel<<<1, 1>>>(dMat, d_in, d_out);
    }, 1);

    test_kernel("Par 1", [&]() {
        dim3 dimGrid{ceilx((double)ROWS/BLOCK_SIZE)};
        dim3 dimBlock{BLOCK_SIZE};
        par_mult_kernel1<<<dimGrid, dimBlock>>>(dMat, d_in, d_out);
    });

    test_kernel("Par 2", [&]() {
        dim3 dimGrid{ROWS};
        dim3 dimBlock{BLOCK_SIZE};
        par_mult_kernel2<<<dimGrid, dimBlock>>>(dMat, d_in, d_out);
    });

    mat.device_free(dMat);
}

#endif // EJERCICIO1_CUH__
