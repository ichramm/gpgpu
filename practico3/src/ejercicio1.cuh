#ifndef EJERCICIO1_CUH__
#define EJERCICIO1_CUH__

#include "practico3.h"

__global__ void kernel_ejercicio1(value_type *mem,
                                  value_type h,
                                  unsigned int N,
                                  unsigned int width)
{
    // 2D grid of 1D block
    auto blockId  = blockIdx.y * gridDim.x + blockIdx.x;
	auto threadId = blockId * blockDim.x + threadIdx.x;

    if (threadId < N) {
        auto y_coord = threadId / width;
        auto x_coord = threadId % width;
        mem[threadId] = f1(COORDVALX(x_coord, h), COORDVALY(y_coord, h));
    }
}

void ejercicio1(value_type **pMatrix = nullptr,
                unsigned int *pN = nullptr,
                unsigned int *pWidth = nullptr) {
    constexpr value_type H = 0.001;
    constexpr unsigned int width = L/H; // truncated to 6283
    constexpr auto N = width * width;

    value_type *d_matrix;
    CUDA_CHK(cudaMalloc(&d_matrix, N * sizeof(value_type))); // no need to cudaMemset

    // 2D grid of 1D blocks
    constexpr auto dimGrid = dim3(ceilx((double)width/BLOCK_SIZE), width, 1);
    constexpr auto dimBlock = dim3(BLOCK_SIZE, 1, 1);

    printf("N      : %10d\n", N);
    printf("Threads: %10d\n", dimGrid.x * dimGrid.y * dimBlock.x);

    float elapsedTime;
    CUDA_MEASURE_START();
    kernel_ejercicio1<<<dimGrid, dimBlock>>>(d_matrix, H, N, width);
    CUDA_CHK(cudaGetLastError());
    CUDA_CHK(cudaDeviceSynchronize());
    CUDA_MEASURE_STOP(elapsedTime);

    value_type *h_matrix;
    CUDA_CHK(cudaMallocHost(&h_matrix, N * sizeof(value_type)));
    CUDA_CHK(cudaMemcpy(h_matrix, d_matrix, N * sizeof(value_type), cudaMemcpyDeviceToHost));

    // hack for reuse
    if (pMatrix) {
        *pMatrix = d_matrix;
        *pN = N;
        *pWidth = width;
        return;
    }

    /* validation */

    auto error_count = 0u;
    for (auto y : fibonacci_numbers) {
        for (auto x : fibonacci_numbers) {
            auto valx = COORDVALX(x, H);
            auto valy = COORDVALY(y, H);
            value_type expected = f1(valx, valy);
            value_type actual = h_matrix[COORD2IDX(x, y, width)];
            if (fabs(expected - actual) > 1e-6) {
                ++error_count;
                printf("(%4d, %4d) Expected: %10.6f, Actual %10.6f, Diff: %10.6f\n",
                        x, y, expected, actual, fabs(expected - actual));
            }
        }
    }

    if (error_count) {
        printf("Detected %d errors\n", error_count);
    } else {
        printf("No errors detected\n");
        printf("Some results:\n");
        for (auto y : fibonacci_numbers) {
            if (y > width || y > 1000) break;
            if (!(y & 1)) continue; // skip some
            for (auto x : fibonacci_numbers) {
                if (x > width || x > 1000) break;
                if (x & 1) continue; // skip some
                printf("   (%3d, %d): %s", y, x, val2string(h_matrix[COORD2IDX(x, y, width)], "% 2.4f", 8).c_str());
            }
            printf("\n");
        }
    }

    CUDA_CHK(cudaFree(d_matrix));
    CUDA_CHK(cudaFreeHost(h_matrix));

    printf("Kernel Elapsed time: %f ms\n", elapsedTime);
}

#endif // EJERCICIO1_CUH__
