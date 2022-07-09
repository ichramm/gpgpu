/*!
 * \file ejercicio1.cu
 * \author Juan Ramirez (juan.ramirez@fing.edu.uy)
 */
#include <iostream>
#include <algorithm>
#include <functional>

#include "lab.hpp"
#include "csr_matrix.hpp"

/**
 * Serial SPMV algorithm. Runs in the Host.
 */
__host__ void ser_spmv_kernel_host(const CSRMatrix<value_type>& mat,
                                   const value_type * __restrict__ x,
                                      value_type *y)
{
    for (auto i = 0u; i < mat.rows; ++i) {
        value_type acc  =0;
        for (auto j = mat.row_pointers[i], end = mat.row_pointers[i+1]; j < end; ++j) {
            acc += mat.values[j] * x[mat.col_indices[j]];
        }
        y[i] = acc;
    }
}

/**
 * Serial SPMV algorithm. Runs in the Device.
 */
__global__ void ser_spmv_kernel_device(CSRMatrix<value_type>::DeviceStruct mat,
                                       const value_type * __restrict__ x,
                                       value_type *y)
{
    for (auto i = 0u; i < mat.rows; ++i) {
        value_type acc  =0;
        for (auto j = mat.row_pointers[i], end = mat.row_pointers[i+1]; j < end; ++j) {
            acc += mat.values[j] * x[mat.col_indices[j]];
        }
        y[i] = acc;
    }
}

/**
 * Parallel SPMV. One thread per row, no shared memory.
 */
__global__ void par_spmv_kernel1(CSRMatrix<value_type>::DeviceStruct mat,
                                 const value_type * __restrict__ x,
                                 value_type *y)
{
    // 1D grid of 1D blocks
    auto row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < mat.rows) {
        value_type acc  =0;

        for (auto j = mat.row_pointers[row], end = mat.row_pointers[row+1]; j < end; ++j) {
            acc += mat.values[j] * x[mat.col_indices[j]];
        }

        y[row] = acc;
    }
}

/**
 * Parallel SPMV. One block per row.
 * - Uses shared memory to store accumulator.
 * - Uses atomic operations to update the accumulator.
 */
__global__ void par_spmv_kernel2(CSRMatrix<value_type>::DeviceStruct mat,
                                 const value_type * __restrict__ x,
                                 value_type *y)
{
    __shared__ value_type row_total;

    if (threadIdx.x == 0) {
        row_total = 0;
    }

    __syncthreads();

    auto row = blockIdx.x;

    if (row < mat.rows) {
        value_type acc = 0;
        for (auto j = mat.row_pointers[row]+threadIdx.x, end = mat.row_pointers[row+1]; j < end; j += blockDim.x) {
            acc += mat.values[j] * x[mat.col_indices[j]];
        }
        atomicAdd(&row_total, acc);
    }

    __syncthreads();

    if (threadIdx.x == 0) {
        y[row] = row_total;
    }
}

/**
 * Parallel SPMV. One block per row.
 * - Uses shared memory to store partial sums.
 * - One accumulator per thread
 * - Performs reduce sum on shared memory
 */
__global__ void par_spmv_kernel3(CSRMatrix<value_type>::DeviceStruct mat,
                                 const value_type * __restrict__ x,
                                 value_type *y)
{
    __shared__ value_type partials[BLOCK_SIZE];

    uint32_t row = blockIdx.x;

    if (row < mat.rows) {
        value_type acc = 0;
        for (uint32_t j = mat.row_pointers[row]+threadIdx.x, end = mat.row_pointers[row+1]; j < end; j += blockDim.x) {
            acc += mat.values[j] * x[mat.col_indices[j]];
        }
        partials[threadIdx.x] = acc;
    }

    __syncthreads();

    for (int offset = BLOCK_SIZE /  2; offset > 0; offset  /= 2) {
        if (threadIdx.x < offset) {
            partials[threadIdx.x] += partials[threadIdx.x + offset];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        y[row] = partials[threadIdx.x];
    }
}

/**
 * Parallel SPMV. One warp per row.
 * - No shared memory
 * - Parallel reduction using __shfl_down
 */
__global__ void par_spmv_kernel4(CSRMatrix<value_type>::DeviceStruct mat,
                                 const value_type * __restrict__ x,
                                 value_type *y)
{
    uint32_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t row = thread_id / 32; // warpSize
    uint32_t lane = thread_id & 31u;

    if (row < mat.rows) {
        value_type acc = 0;

        for (uint32_t j = mat.row_pointers[row]+lane, end = mat.row_pointers[row+1]; j < end; j += 32) {
            acc += mat.values[j] * x[mat.col_indices[j]];
        }

        __syncthreads();

        for (int offset = warpSize/2; offset > 0; offset /= 2) {
#if __CUDA_ARCH__ >= 300
            acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
#else
            acc += __shfl_down(acc, offset);
#endif
        }

        if (lane == 0) {
            y[row] = acc;
        }
    }
}

static void initial_kindergarten_test()
{
    constexpr uint32_t rows = 6;
    constexpr uint32_t columns = 5;

    std::vector<value_type> h_vals{{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}};
    std::vector<uint32_t> h_colIdx{{0, 1, 2, 4, 1, 2, 3, 4, 3, 1, 4}};
    std::vector<uint32_t> h_rowPtr{{0, 1, 4, 4, 8, 9, 11}}; // N+1 elements (last is out of vals's bounds)
    value_type h_vecX[columns] = {1, 2, 1, 1, 1};
    value_type h_expectedResult[rows] = {1, 11, 0, 31, 9, 31}; // column sum

    CSRMatrix<value_type> mat{6u, 5u, h_vals, h_colIdx, h_rowPtr};

    std::cout << "Matrix:" << std::endl;
    std::cout << mat << std::endl;

    auto d_vecX = dev_alloc_fill(columns, h_vecX);
    auto d_vecY = dev_alloc_zero<value_type>(rows);
    auto d_expectedResult = dev_alloc_fill(rows, h_expectedResult);

    auto dMat = mat.to_device();

    ser_spmv_kernel_device<<<1, 1>>>(dMat, d_vecX.get(), d_vecY.get());
    CUDA_CHK(cudaGetLastError());
    CUDA_CHK(cudaDeviceSynchronize());
    validate_results("ser_spmv_kernel_device", d_expectedResult.get(), d_vecY.get(), rows);

    cudaMemset(d_vecY.get(), 0, sizeof(value_type)*rows);
    par_spmv_kernel1<<<1, rows>>>(dMat, d_vecX.get(), d_vecY.get());
    CUDA_CHK(cudaGetLastError());
    CUDA_CHK(cudaDeviceSynchronize());
    validate_results("par_spmv_kernel1", d_expectedResult.get(), d_vecY.get(), rows);

    cudaMemset(d_vecY.get(), 0, sizeof(value_type)*rows);
    par_spmv_kernel2<<<rows, 32>>>(dMat, d_vecX.get(), d_vecY.get());
    CUDA_CHK(cudaGetLastError());
    CUDA_CHK(cudaDeviceSynchronize());
    validate_results("par_spmv_kernel2", d_expectedResult.get(), d_vecY.get(), rows);

    cudaMemset(d_vecY.get(), 0, sizeof(value_type)*rows);
    par_spmv_kernel3<<<rows, 32>>>(dMat, d_vecX.get(), d_vecY.get());
    CUDA_CHK(cudaGetLastError());
    CUDA_CHK(cudaDeviceSynchronize());
    validate_results("par_spmv_kernel3", d_expectedResult.get(), d_vecY.get(), rows);

    cudaMemset(d_vecY.get(), 0, sizeof(value_type)*rows);
    par_spmv_kernel4<<<rows, 32>>>(dMat, d_vecX.get(), d_vecY.get());
    CUDA_CHK(cudaGetLastError());
    CUDA_CHK(cudaDeviceSynchronize());
    validate_results("par_spmv_kernel4", d_expectedResult.get(), d_vecY.get(), rows);

    mat.device_free(dMat);
}

void ejercicio1() {
    initial_kindergarten_test();

    CSRMatrix<value_type> mat{MAT_ROWS, MAT_COLS};
    mat.random_init();

    value_type h_vecX[MAT_COLS];
    value_type h_vecY[MAT_ROWS] = {0};
    std::fill(std::begin(h_vecX), std::end(h_vecX), 1);

    auto d_vecX = dev_alloc_fill(MAT_COLS, h_vecX);
    auto d_vecY = dev_alloc_zero<value_type>(MAT_ROWS);
    auto d_expectedResult = dev_alloc<value_type>(MAT_ROWS);

    auto dMat = mat.to_device();

    auto run_kernel = [&](std::string name,
                          std::function<void()> f,
                          value_type *d_expectedResult = nullptr,
                          int times = BENCH_TIMES) {
        std::cout << name << ": " << std::endl;
        Metric m;
        for (auto i = 0; i < times; ++i) {
            CUDA_CHK(cudaMemset(d_vecY.get(), 0, sizeof(value_type)*MAT_ROWS));
            auto t = m.track_begin();
            f();
            CUDA_CHK(cudaGetLastError());
            CUDA_CHK(cudaDeviceSynchronize());
            m.track_end(t);
            if (d_expectedResult) {
                if (!validate_results(name.c_str(), d_expectedResult, d_vecY.get(), MAT_ROWS, true, false)) {
                    break;
                }
            }
        }
        printf("> %f ms, stdev: %f, CV: %f\n", m.mean(), m.stdev(), m.cv());
    };

    run_kernel("ser_spmv_kernel_device", [&]() {
        ser_spmv_kernel_device<<<1, 1>>>(dMat, d_vecX.get(), d_vecY.get());
    }, nullptr, 1);

    // assume result of serial operation as ground truth
    CUDA_CHK(cudaMemcpy(d_expectedResult.get(), d_vecY.get(), sizeof(value_type)*MAT_ROWS, cudaMemcpyDeviceToDevice));

    run_kernel("ser_spmv_kernel_host", [&]() {
        ser_spmv_kernel_host(mat, h_vecX, h_vecY);
    }, nullptr, 1);

    { // using h_vecY so it doesn't get optimized away
        cudaMemcpy(d_vecY.get(), h_vecY, 1, cudaMemcpyHostToDevice);
    }

    run_kernel("par_spmv_kernel1", [&]() {
        dim3 dimGrid{ceilx((double)MAT_ROWS/BLOCK_SIZE)};
        dim3 dimBlock{BLOCK_SIZE};
        par_spmv_kernel1<<<dimGrid, dimBlock>>>(dMat, d_vecX.get(), d_vecY.get());
    }, d_expectedResult.get());

    run_kernel("par_spmv_kernel2", [&]() {
        dim3 dimGrid{MAT_ROWS};
        dim3 dimBlock{BLOCK_SIZE}; // works better with fewer threads
        par_spmv_kernel2<<<dimGrid, dimBlock>>>(dMat, d_vecX.get(), d_vecY.get());
    }, d_expectedResult.get());

    run_kernel("par_spmv_kernel3", [&]() {
        dim3 dimGrid{MAT_ROWS};
        dim3 dimBlock{BLOCK_SIZE};
        par_spmv_kernel3<<<dimGrid, dimBlock>>>(dMat, d_vecX.get(), d_vecY.get());
    }, d_expectedResult.get());

    run_kernel("par_spmv_kernel4", [&]() {
        dim3 dimGrid{MAT_ROWS/4};
        dim3 dimBlock{BLOCK_SIZE};
        par_spmv_kernel4<<<dimGrid, dimBlock>>>(dMat, d_vecX.get(), d_vecY.get());
    }, d_expectedResult.get());

    mat.device_free(dMat);
}
