#ifndef EJERCICIO1_CUH__
#define EJERCICIO1_CUH__

#include "practico3.h"

__global__ void kernel_ejercicio1(value_type *mem,
                                  value_type h,
                                  unsigned int N,
                                  unsigned int width,
                                  unsigned int mem_xdim)  // same as width if there's no padding
{
    // 2D grid of 1D block
    auto blockId  = blockIdx.y * gridDim.x + blockIdx.x;
	auto threadId = blockId * blockDim.x + threadIdx.x;

    if (threadId < N) {
        auto y_coord = threadId / mem_xdim;
        auto x_coord = threadId % mem_xdim;
        if (x_coord < width) {  // always true if there's no padding
            mem[threadId] = f1(COORDVALX(x_coord, h), COORDVALY(y_coord, h));
        }
    }
}

void ejercicio1() {
    constexpr value_type H = 0.001;
    constexpr unsigned int width = L/H; // truncated to 6283

    constexpr auto mem_ydim = width;
#ifdef DISABLE_PADDING
    constexpr auto mem_xdim = width;
#else
    // for full coalesced access the width must be a multiple of the warp size (doesn't help much anyway)
    constexpr auto mem_xdim = roundup_to_warp_size(width);
#endif

    constexpr auto N = mem_ydim * mem_xdim;

    value_type *d_matrix;
    cudaMalloc(&d_matrix, N * sizeof(value_type)); // no need to cudaMemset

    // 2D grid of 1D blocks
    auto dimGrid = dim3(ceilx((double)mem_xdim/BLOCK_SIZE), mem_ydim, 1);
    auto dimBlock = dim3(BLOCK_SIZE, 1, 1);

    printf("N      : %10d\n", N);
    printf("Threads: %10d\n", dimGrid.x * dimGrid.y * dimBlock.x);

    float elapsedTime;
    CUDA_MEASURE_START();
    kernel_ejercicio1<<<dimGrid, dimBlock>>>(d_matrix, H, N, width, mem_xdim);
    CUDA_CHK(cudaGetLastError());
    CUDA_CHK(cudaDeviceSynchronize());
    CUDA_MEASURE_STOP(elapsedTime);
    printf("Elapsed time: %f ms\n", elapsedTime);

    value_type *h_matrix;
    CUDA_CHK(cudaMallocHost(&h_matrix, N * sizeof(value_type)));
    CUDA_CHK(cudaMemcpy(h_matrix, d_matrix, N * sizeof(value_type), cudaMemcpyDeviceToHost));

    /* validation */

    auto error_count = 0u;
    for (auto y : fibonacci_numbers) {
        for (auto x : fibonacci_numbers) {
            auto valx = COORDVALX(x, H);
            auto valy = COORDVALY(y, H);
            value_type expected = f1(valx, valy);
            value_type actual = h_matrix[COORD2IDX(x, y, mem_xdim)];
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
    }

    cudaFree(d_matrix);
    cudaFreeHost(h_matrix);
}

#endif // EJERCICIO1_CUH__
