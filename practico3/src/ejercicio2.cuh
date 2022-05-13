#ifndef EJERCICIO2_CUH__
#define EJERCICIO2_CUH__

#include "practico3.h"

__global__ void kernel_ejercicio2(value_type *mem,
                                  value_type h,
                                  unsigned int N,
                                  unsigned int width,
                                  unsigned int mem_xdim, // same as width if there's no padding
                                  unsigned int mem_ydim)
{
    // 3D grid of 1D block
    auto blockId  = blockIdx.z * (gridDim.y * gridDim.x) + blockIdx.y * gridDim.x + blockIdx.x;
    auto threadId = blockId * blockDim.x + threadIdx.x;

    if (threadId < N) {
        auto z_coord = threadId / (mem_ydim * mem_xdim);
        auto y_coord = (threadId % (mem_ydim * mem_xdim)) / mem_xdim;
        auto x_coord = threadId % mem_xdim;
        if (x_coord < width) {  // always true if there's no padding
            mem[threadId] = f2(COORDVALX(x_coord, h),
                               COORDVALY(y_coord, h),
                               COORDVALZ(z_coord, h));
        }
    }
}

void ejercicio2(value_type **pCube = nullptr,
                unsigned int *pN = nullptr,
                unsigned int *pWidth = nullptr,
                unsigned int *pMemXdim = nullptr,
                unsigned int *pMemYdim = nullptr,
                unsigned int *pMemZdim = nullptr) {
    constexpr value_type H = 0.01;
    constexpr unsigned int width = L/H; // truncated to 628

    constexpr auto mem_zdim = width;
    constexpr auto mem_ydim = width;
#ifdef DISABLE_PADDING
    constexpr auto mem_xdim = width;
#else
    // for full coalesced access the width must be a multiple of the warp size (doesn't help much anyway)
    constexpr auto mem_xdim = roundup_to_warp_size(width);
#endif

    constexpr auto N = mem_zdim * mem_ydim * mem_xdim;

    value_type *d_cube;
    cudaMalloc(&d_cube, N * sizeof(value_type)); // no need to cudaMemset

    // 3D grid of 1D blocks
    constexpr auto dimGrid = dim3(ceilx((double)mem_xdim/BLOCK_SIZE), mem_ydim, mem_zdim);
    constexpr auto dimBlock = dim3(BLOCK_SIZE, 1, 1);

    printf("N      : %10d\n", N);
    printf("Threads: %10d\n", dimGrid.x * dimGrid.y * dimGrid.z * dimBlock.x);

    float elapsedTime;
    CUDA_MEASURE_START();
    kernel_ejercicio2<<<dimGrid, dimBlock>>>(d_cube, H, N, width, mem_xdim, mem_ydim);
    CUDA_CHK(cudaGetLastError());
    CUDA_CHK(cudaDeviceSynchronize());
    CUDA_MEASURE_STOP(elapsedTime);

    if (pCube) {
        *pCube = d_cube;
        *pN = N;
        *pWidth = width;
        *pMemXdim = mem_xdim;
        *pMemYdim = mem_ydim;
        *pMemZdim = mem_zdim;
        return;
    }

    /* validation */

    value_type *h_cube;
    CUDA_CHK(cudaMallocHost(&h_cube, N * sizeof(value_type)));
    CUDA_CHK(cudaMemcpy(h_cube, d_cube, N * sizeof(value_type), cudaMemcpyDeviceToHost));

    auto error_count = 0;
    for (auto z : fibonacci_numbers) {
        if (z > width)
            break;
        for (auto y : fibonacci_numbers) {
            if (y > width)
                break;
            for (auto x : fibonacci_numbers) {
                if (x > width)
                    break;
                auto expected = f2(COORDVALX(x, H), COORDVALY(y, H), COORDVALZ(z, H));
                auto actual = h_cube[COORD3IDX(x, y, z, mem_xdim, mem_ydim)];
                if (fabs(expected - actual) > 1e-6) {
                    ++error_count;
                    printf("(%4d, %4d, %4d) Expected: %10.6f, Actual %10.6f, Diff: %10.6f\n",
                           x, y, z, expected, actual, fabs(expected - actual));
                }
            }
        }
    }

    if (error_count) {
        printf("Detected %d errors\n", error_count);
    } else {
        printf("No errors detected\n");
    }

    cudaFree(d_cube);
    cudaFreeHost(h_cube);

    printf("Elapsed time: %f ms\n", elapsedTime);
}

#endif // EJERCICIO2_CUH__
