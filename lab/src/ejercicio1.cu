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
__host__ void serial_spmv_kernel_host(const CSRMatrix<value_type>& mat,
                                      value_type *x,
                                      value_type *y) {
    for (uint32_t i = 0; i < mat.rows; ++i) {
        value_type acc  =0;
        for (uint32_t j = mat.row_pointers[i], end = mat.row_pointers[i+1]; j < end; ++j) {
            acc += mat.values[j] * x[mat.col_indices[j]];
        }
        y[i] = acc;
    }
}

/**
 * Serial SPMV algorithm. Runs in the Device.
 */
__global__ void serial_spmv_kernel_device(CSRMatrix<value_type>::DeviceStruct mat,
                                          value_type *x,
                                          value_type *y) { // already zeroed
    for (uint32_t i = 0; i < mat.rows; ++i) {
        value_type acc  =0;
        for (uint32_t j = mat.row_pointers[i], end = mat.row_pointers[i+1]; j < end; ++j) {
            acc += mat.values[j] * x[mat.col_indices[j]];
        }
        y[i] = acc;
    }
}

/**
 * Parallel SPMV. One thread per row, no shared memory.
 */
__global__ void par_spmv_kernel1(CSRMatrix<value_type>::DeviceStruct mat,
                                 value_type *in,
                                 value_type *out) {
    // 1D grid of 1D blocks
    uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < mat.rows) {
        value_type acc  =0;

        for (uint32_t j = mat.row_pointers[row], end = mat.row_pointers[row+1]; j < end; ++j) {
            acc += mat.values[j] * in[mat.col_indices[j]];
        }

        out[row] = acc;
    }
}

/**
 * Parallel SPMV. One block per row.
 * - Uses shared memory to store accumulator.
 * - Uses atomic operations to update the accumulator.
 */
__global__ void par_spmv_kernel2(CSRMatrix<value_type>::DeviceStruct mat,
                                 value_type *in,
                                 value_type *out) {
    __shared__ value_type row_total;

    if (threadIdx.x == 0) {
        row_total = 0;
    }

    __syncthreads();

    uint32_t row = blockIdx.x;

    if (row < mat.rows) {
        value_type acc = 0;
        for (uint32_t j = mat.row_pointers[row]+threadIdx.x, end = mat.row_pointers[row+1]; j < end; j += blockDim.x) {
            acc += mat.values[j] * in[mat.col_indices[j]];
        }
        atomicAdd(&row_total, acc);
    }

    __syncthreads();

    if (threadIdx.x == 0) {
        out[row] = row_total;
    }
}

/**
 * Parallel SPMV. One block per row.
 * - Uses shared memory to store partial sums.
 * - One accumulator per thread
 * - Performs reduce sum on shared memory
 */
__global__ void par_spmv_kernel3(CSRMatrix<value_type>::DeviceStruct mat,
                                 value_type *in,
                                 value_type *out) {
    __shared__ value_type partials[BLOCK_SIZE];

    uint32_t row = blockIdx.x;

    if (row < mat.rows) {
        value_type acc = 0;
        for (uint32_t j = mat.row_pointers[row]+threadIdx.x, end = mat.row_pointers[row+1]; j < end; j += blockDim.x) {
            acc += mat.values[j] * in[mat.col_indices[j]];
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
        out[row] = partials[threadIdx.x];
    }
}

/**
 * Parallel SPMV. One warp per row.
 * - No shared memory
 * - Parallel reduction using __shfl_down
 */
__global__ void par_spmv_kernel4(CSRMatrix<value_type>::DeviceStruct mat,
                                 value_type *in,
                                 value_type *out) {
    uint32_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t row = thread_id / 32; // warpSize
    uint32_t lane = thread_id & 31u;

    if (row < mat.rows) {
        value_type acc = 0;

        for (uint32_t j = mat.row_pointers[row]+lane, end = mat.row_pointers[row+1]; j < end; j += 32) {
            acc += mat.values[j] * in[mat.col_indices[j]];
        }

        __syncthreads();

        for (int offset = warpSize/2; offset > 0; offset /= 2) {
            acc += __shfl_down(acc, offset);
        }

        if (lane == 0) {
            out[row] = acc;
        }
    }
}

bool validate_results(value_type *d_expectedResult,
                      value_type *d_out,
                      uint32_t N,
                      bool silent = false,
                      bool print_vect = true) {
    if (auto ndiff = gpu_compare_arrays(d_expectedResult, d_out, N)) {
        value_type *h_out = new value_type[N];
        cudaMemcpy(h_out, d_out, sizeof(h_out), cudaMemcpyDeviceToHost);
        std::cout << "Error: " << ndiff << " differences found" << std::endl;
        if (print_vect) {
            for (uint32_t i = 0; i < N; ++i) {
                std::cout << " " << h_out[i];
            }
            std::cout << std::endl;
        }
        delete[] h_out;
        return false;
    } else {
        if (!silent) {
            std::cout << "OK" << std::endl;
        }
        return true;
    }
}


void initial_algorithm_test() {
    constexpr uint32_t N = 6;
    constexpr uint32_t C = 5;

    std::vector<value_type> h_vals{{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}};
    std::vector<uint32_t> h_colIdx{{0, 1, 2, 4, 1, 2, 3, 4, 3, 1, 4}};
    std::vector<uint32_t> h_rowPtr{{0, 1, 4, 4, 8, 9, 11}}; // N+1 elements (last is out of vals's bounds)
    value_type h_in[C] = {1, 1, 1, 1, 1};
    value_type h_expectedResult[N] = {1, 9, 0, 26, 9, 21}; // column sum

    value_type *d_in, *d_out, *d_expectedResult;

    CSRMatrix<value_type> mat{6u, 5u, h_vals, h_colIdx, h_rowPtr};

    std::cout << "Matrix:" << std::endl;
    std::cout << mat << std::endl;

    CUDA_CHK(cudaMalloc(&d_in, sizeof(h_in)));
    CUDA_CHK(cudaMemcpy(d_in, h_in, sizeof(h_in), cudaMemcpyHostToDevice));
    CUDA_CHK(cudaMalloc(&d_out, sizeof(value_type)*N));
    CUDA_CHK(cudaMemset(d_out, 0, sizeof(value_type)*N));

    CUDA_CHK(cudaMalloc(&d_expectedResult, sizeof(h_expectedResult)));
    CUDA_CHK(cudaMemcpy(d_expectedResult, h_expectedResult, sizeof(h_expectedResult), cudaMemcpyHostToDevice));

    auto dMat = mat.to_device();

    cudaMemset(d_out, 0, sizeof(value_type)*N);
    serial_spmv_kernel_device<<<1, 1>>>(dMat, d_in, d_out);
    CUDA_CHK(cudaGetLastError());
    CUDA_CHK(cudaDeviceSynchronize());
    validate_results(d_expectedResult, d_out, N);

    cudaMemset(d_out, 0, sizeof(value_type)*N);
    par_spmv_kernel1<<<1, N>>>(dMat, d_in, d_out);
    CUDA_CHK(cudaGetLastError());
    CUDA_CHK(cudaDeviceSynchronize());
    validate_results(d_expectedResult, d_out, N);

    cudaMemset(d_out, 0, sizeof(value_type)*N);
    par_spmv_kernel2<<<N, 32>>>(dMat, d_in, d_out);
    CUDA_CHK(cudaGetLastError());
    CUDA_CHK(cudaDeviceSynchronize());
    validate_results(d_expectedResult, d_out, N);

    cudaMemset(d_out, 0, sizeof(value_type)*N);
    par_spmv_kernel3<<<N, 32>>>(dMat, d_in, d_out);
    CUDA_CHK(cudaGetLastError());
    CUDA_CHK(cudaDeviceSynchronize());
    validate_results(d_expectedResult, d_out, N);

    cudaMemset(d_out, 0, sizeof(value_type)*N);
    par_spmv_kernel4<<<N, 32>>>(dMat, d_in, d_out);
    CUDA_CHK(cudaGetLastError());
    CUDA_CHK(cudaDeviceSynchronize());
    validate_results(d_expectedResult, d_out, N);

    mat.device_free(dMat);
}

void ejercicio1() {
    initial_algorithm_test();

    const unsigned int ROWS = 10000u;
    const unsigned int COLUMNS = 9000u;

    CSRMatrix<value_type> mat{ROWS, COLUMNS};
    mat.random_init();

    value_type h_in[COLUMNS];
    value_type h_out[ROWS] = {0};
    std::fill(std::begin(h_in), std::end(h_in), 1);

    value_type *d_in, *d_out, *d_expectedResult;

    CUDA_CHK(cudaMalloc(&d_in, sizeof(value_type)*COLUMNS));
    CUDA_CHK(cudaMalloc(&d_out, sizeof(value_type)*ROWS));

    CUDA_CHK(cudaMemcpy(d_in, h_in, sizeof(value_type)*COLUMNS, cudaMemcpyHostToDevice));

    auto dMat = mat.to_device();

    auto run_kernel = [&](std::string name,
                          std::function<void()> f,
                          value_type *d_expectedResult = nullptr,
                          int times = BENCH_TIMES) {
        std::cout << name << ": " << std::endl;
        Metric m;
        for (auto i = 0; i < times; ++i) {
            CUDA_CHK(cudaMemset(d_out, 0, sizeof(value_type)*ROWS));
            auto t = m.track_begin();
            f();
            CUDA_CHK(cudaGetLastError());
            CUDA_CHK(cudaDeviceSynchronize());
            m.track_end(t);
            if (d_expectedResult) {
                if (!validate_results(d_expectedResult, d_out, ROWS, true, false)) {
                    break;
                }
            }
        }
        printf("> %f ms, stdev: %f, CV: %f\n", m.mean(), m.stdev(), m.cv());
    };

    run_kernel("Serial (GPU)", [&]() {
        serial_spmv_kernel_device<<<1, 1>>>(dMat, d_in, d_out);
    }, nullptr, 1);

    // assume result of serial operatin as ground truth
    CUDA_CHK(cudaMalloc(&d_expectedResult, sizeof(value_type)*ROWS));
    CUDA_CHK(cudaMemcpy(d_expectedResult, d_out, sizeof(value_type)*ROWS, cudaMemcpyDeviceToDevice));

    run_kernel("Serial (CPU)", [&]() {
        serial_spmv_kernel_host(mat, h_in, h_out);
    }, nullptr, 1);

    { // using h_out so it doesn't get optimized away
        cudaMemcpy(d_out, h_out, 1, cudaMemcpyHostToDevice);
    }

    run_kernel("Par 1", [&]() {
        dim3 dimGrid{ceilx((double)ROWS/BLOCK_SIZE)};
        dim3 dimBlock{BLOCK_SIZE};
        par_spmv_kernel1<<<dimGrid, dimBlock>>>(dMat, d_in, d_out);
    }, d_expectedResult);

    run_kernel("Par 2", [&]() {
        dim3 dimGrid{ROWS};
        dim3 dimBlock{BLOCK_SIZE}; // works better with fewer threads
        par_spmv_kernel2<<<dimGrid, dimBlock>>>(dMat, d_in, d_out);
    }, d_expectedResult);

    run_kernel("Par 3", [&]() {
        dim3 dimGrid{ROWS};
        dim3 dimBlock{BLOCK_SIZE};
        par_spmv_kernel3<<<dimGrid, dimBlock>>>(dMat, d_in, d_out);
    }, d_expectedResult);

    run_kernel("Par 4", [&]() {
        dim3 dimGrid{ROWS/4};
        dim3 dimBlock{BLOCK_SIZE};
        par_spmv_kernel4<<<dimGrid, dimBlock>>>(dMat, d_in, d_out);
    }, d_expectedResult);

    mat.device_free(dMat);

    CUDA_CHK(cudaFree(d_out));
    CUDA_CHK(cudaFree(d_in));
}
