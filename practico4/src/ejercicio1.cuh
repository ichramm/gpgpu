/*!
 * \file ejercicio1.cuh
 * \author Juan Ramirez (juan.ramirez@fing.edu.uy)
 */
#ifndef EJERCICIO1_CUH__
#define EJERCICIO1_CUH__

#include "practico4.h"
#include <iostream>
#include <vector>
#include <numeric>
#include <cstring>


// constantes necesarias del práctico 2
static constexpr int M = 256;


/*!
 * \brief Kernel del ejercicio del práctico 2
 */
__global__ void count_occurrences(const int* __restrict__ d_message,
                                  unsigned int length,
                                  unsigned int *d_counts)
{
    auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < length) {
        atomicAdd(&(d_counts[d_message[idx]]), 1);
    }
}

/*!
 * \brief Kernel del ejercicio del práctico 2 que hace uso de memoria compartida.
 */
__global__ void count_occurrences_shm(const int* __restrict__ d_message,
                                      unsigned int length,
                                      unsigned int *d_counts)
{
    __shared__ unsigned int partial_counts[M];

    // init shared memory
    for (auto i = threadIdx.x; i < M; i += blockDim.x) {
        partial_counts[i] = 0;
    }

    __syncthreads();

    auto msg_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (msg_idx < length) {
        atomicAdd(&(partial_counts[d_message[msg_idx]]), 1);
    }

    __syncthreads();

    if (threadIdx.x == 0) {
        for (int i = 0; i < M; ++i) {
            atomicAdd(&(d_counts[i]), partial_counts[i]);
        }
    }
}

__global__ void count_occurrences_shm_stride(const int* __restrict__ d_message,
                                             unsigned int length,
                                             unsigned int *d_counts)
{
    __shared__ unsigned int partial_counts[M];

    // init shared memory
    for (auto i = threadIdx.x; i < M; i += blockDim.x) {
        partial_counts[i] = 0;
    }

    __syncthreads();

    auto msg_idx = blockIdx.x * blockDim.x + threadIdx.x;

    for (; msg_idx < length; msg_idx += blockDim.x*gridDim.x) {
        auto msg = d_message[msg_idx];
        atomicAdd(&(partial_counts[msg]), 1);
    }

    __syncthreads();

    if (threadIdx.x == 0) {
        for (int i = 0; i < M; ++i) {
            atomicAdd(&(d_counts[i]), partial_counts[i]);
        }
    }
}


static void print_counts(unsigned int *h_counts)
{
    int printed = 0;
    for (int i = 0; i < M; ++i) {
        if (h_counts[i] > 0) {
            printf(printed ? ",  %3d: %6d" : " %3d: %6d", i, h_counts[i]);
            ++printed;
            if (printed > 8) {
                printf("\n");
                printed = 0;
            }
        }
    }
    if (printed > 0) {
        printf("\n");
    }
}

static void print_offending_counts(unsigned int *counts,
                                   unsigned int *ground_truth,
                                   unsigned int sum_total,
                                   unsigned int msg_length)
{
    printf("total: %u, expected: %u\n", sum_total, msg_length);
    for (int i = 0; i < M; ++i) {
        if (counts[i] != ground_truth[i]) {
            printf("%#04x: got %6u, expected: %6u\n", i, counts[i], ground_truth[i]);
        }
    }
}


static void process_results(const char *test_name,
                            unsigned int block_size,
                            Metric metric,
                            unsigned int msg_length,
                            unsigned int *d_counts,
                            unsigned int *h_counts,
                            bool validate = false)
{
    auto mean = metric.mean();
    auto stdev = metric.stdev();
    printf("%s (BLSZ=%u) - mean: %f ms, stdev: %f, CV: %f\n", test_name, block_size, mean, stdev, stdev/mean);

    // tricky: if validate = true, then h_counts contains the ground truth
    if (validate) {
        // copy to local array and compare to h_counts
        unsigned int temp_counts[M];
        cudaMemcpy(temp_counts, d_counts, M * sizeof(unsigned int), cudaMemcpyDeviceToHost);
        auto total = std::accumulate(temp_counts, temp_counts+M, 0u);
        if (total != msg_length || std::memcmp(h_counts, temp_counts, M * sizeof(unsigned int))) {
            printf("\tERROR: counts don't match\n");
            print_offending_counts(temp_counts, h_counts, total, msg_length);
        } else {
            printf("\tOK\n");
        }
    } else {
        // fill h_counts and print
        cudaMemcpy(h_counts, d_counts, M * sizeof(unsigned int), cudaMemcpyDeviceToHost);
        auto total = std::accumulate(h_counts, h_counts+M, 0u);
#ifdef PRINT_OUTPUT
        printf("Output (total: %u):\n", total);
        print_counts(h_counts);
#endif
    }
}


static void ejercicio1(const char *fname)
{
    int *d_message, *h_message;
    unsigned int *d_counts;
    unsigned int h_counts[M] = {0};

    const auto length = read_file(fname, &h_message);
    printf("File size: %u\n", length);

    CUDA_CHK(cudaMalloc(&d_message, length * sizeof(int)));
    CUDA_CHK(cudaMemcpy(d_message, h_message, length * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHK(cudaMalloc(&d_counts, M * sizeof(unsigned int)));

    { // simple kernel
        Metric metric;
        for (auto i = 0; i < 100; ++i) {
            dim3 dimGrid(ceilx((double)length/BLOCK_SIZE), 1, 1);
            dim3 dimBlock(BLOCK_SIZE, 1, 1);
            CUDA_CHK(cudaMemset(d_counts, 0, M * sizeof(unsigned int)));
            auto t = metric.track_begin();
            count_occurrences<<<dimGrid, dimBlock>>>(d_message, length, d_counts);
            CUDA_CHK(cudaGetLastError());
            CUDA_CHK(cudaDeviceSynchronize());
            metric.track_end(t);
        }
        process_results("Simple Kernel", BLOCK_SIZE, metric, length, d_counts, h_counts);
    }

    // note: block-size = 32 was too slow
    constexpr unsigned int block_sizes[] = {64, 128, 256, 512, 1024};

    // shared memory kernel
    for (int block_size : block_sizes) {
        Metric metric;
        for (auto i = 0; i < 100; ++i) {
            dim3 dimGrid(ceilx((double)length/block_size), 1, 1);
            dim3 dimBlock(block_size, 1, 1);
            CUDA_CHK(cudaMemset(d_counts, 0, M * sizeof(unsigned int)));
            auto t = metric.track_begin();
            count_occurrences_shm<<<dimGrid, dimBlock>>>(d_message, length, d_counts);
            CUDA_CHK(cudaGetLastError());
            CUDA_CHK(cudaDeviceSynchronize());
            metric.track_end(t);
        }
        process_results("SHM Kernel", block_size, metric, length, d_counts, h_counts, true);
    }

    // shared memory kernel with stride

    constexpr unsigned int total_threads[] = {8192, 16384, 32768};

    for (auto thread_count : total_threads) {
        std::string test_name = std::string{"Strided SHM Kernel"} + " (threads: " + std::to_string(thread_count) + ")";
        for (auto block_size : block_sizes) {
            Metric metric;
            for (auto i = 0; i < 100; ++i) {
                auto num_blocks = thread_count / block_size;
                dim3 dimGrid(num_blocks, 1, 1);
                dim3 dimBlock(block_size, 1, 1);
                CUDA_CHK(cudaMemset(d_counts, 0, M * sizeof(unsigned int)));
                auto t = metric.track_begin();
                count_occurrences_shm_stride<<<dimGrid, dimBlock>>>(d_message, length, d_counts);
                CUDA_CHK(cudaGetLastError());
                CUDA_CHK(cudaDeviceSynchronize());
                metric.track_end(t);
            }
            process_results(test_name.c_str(), block_size, metric, length, d_counts, h_counts, true);
        }
    }

    cudaFree(d_counts);
    cudaFree(d_message);
    free(h_message);
}

#endif // EJERCICIO1_CUH__
