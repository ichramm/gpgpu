/*!
 * \file ejercicio2.cuh
 * \author Juan Ramirez (juan.ramirez@fing.edu.uy)
 */
#ifndef EJERCICIO2_CUH__
#define EJERCICIO2_CUH__

#include "practico4.h"

#ifndef SHM_BLOCK_WIDTH
#define SHM_BLOCK_WIDTH 16u
#endif

/*!
 * Kernel del practico 3 adaptado a este stencil
 */
__global__ void stencil_kernel_gm(const value_type * __restrict__ in,
                                  value_type *out,
                                  value_type hsq,
                                  uint32_t N,
                                  uint32_t width)
{
    // 2D grid of 2D blocks
    auto g_x = blockIdx.x*blockDim.x + threadIdx.x;
    auto g_y = blockIdx.y*blockDim.y + threadIdx.y;
    auto g_idx = g_y*width + g_x;

    if (g_x < width && g_y < width) {
        auto value = (-8) * in[g_idx] +
                     (g_x < width-1 ? in[g_idx + 1] : 0) +
                     (g_x < width-2 ? in[g_idx + 2] : 0) +
                     (g_x > 0 ? in[g_idx-1] : 0) +
                     (g_x > 1 ? in[g_idx-2] : 0) +
                     (g_idx < N - width ? in[g_idx + width] : 0) +
                     (g_idx < N - (2*width) ? in[g_idx + (2*width)] : 0) +
                     (g_idx >= width ? in[g_idx - width] : 0) +
                     (g_idx >= (2*width) ? in[g_idx - (2*width)] : 0);

        out[g_idx] = value / hsq;
    }
}


/*!
 * Este Kernel utiliza un tile de ancho SHM_BLOCK_WIDTH+4:
 * - Los threads de la 1er y 2da fila cargan el halo superior
 * - Los threads de la 1er y 2da columna cargan el halo izquierdo
 * - Análogamente, los últimos threads cargan el halo inferior y derecho
 */
__global__ void stencil_kernel_shm(const value_type * __restrict__ in,
                                   value_type *out,
                                   value_type hsq,
                                   uint32_t N,
                                   uint32_t width)
{
    __shared__ value_type tile[SHM_BLOCK_WIDTH+4][SHM_BLOCK_WIDTH+4];

    // 2D grid of 2D blocks
    auto g_x = blockIdx.x*blockDim.x + threadIdx.x;
    auto g_y = blockIdx.y*blockDim.y + threadIdx.y;
    auto g_idx = g_y*width + g_x;
    auto shm_x = threadIdx.x + 2;
    auto shm_y = threadIdx.y + 2;

    if (g_idx < N) {
        tile[shm_y][shm_x] = in[g_idx];

        if (threadIdx.x < 2) {
            tile[shm_y][shm_x-2] = g_x > 1 ? in[g_idx-2] : 0;
        }
        else if (threadIdx.x >= blockDim.x - 2) {
            tile[shm_y][shm_x+2] = g_x < width - 2 ? in[g_idx+2] : 0;
        }
        if (threadIdx.y < 2) {
            tile[shm_y-2][shm_x] = g_y > 1 ? in[g_idx-(2*width)] : 0;
        }
        else if (threadIdx.y >= blockDim.y - 2) {
            tile[shm_y+2][shm_x] = g_idx < N - (2 * width) ? in[g_idx+(2*width)] : 0;
        }
    }

    __syncthreads();

    if (g_idx < N) {
        value_type value = tile[shm_y][shm_x] * (-8) +
                           tile[shm_y][shm_x+1] +
                           tile[shm_y][shm_x+2] +
                           tile[shm_y][shm_x-1] +
                           tile[shm_y][shm_x-2] +
                           tile[shm_y+1][shm_x] +
                           tile[shm_y+2][shm_x] +
                           tile[shm_y-1][shm_x] +
                           tile[shm_y-2][shm_x];

        out[g_idx] = value / hsq;
    }
}

void ejercicio2_impl(int32_t width) {
    printf("Matrix Width: %u\n", width);
    constexpr value_type H = 0.001;
    value_type *d_inMatrix, *d_outMatrix1, *d_outMatrix2;
    int32_t N = width*width;

    { // setup
        CUDA_CHK(cudaMalloc(&d_inMatrix, sizeof(value_type) * N));
        CUDA_CHK(cudaMalloc(&d_outMatrix1, sizeof(value_type) * N));
        CUDA_CHK(cudaMalloc(&d_outMatrix2, sizeof(value_type) * N));

        {
            curandGenerator_t prng;
            curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_XORWOW);
            CUDA_CHK(cudaGetLastError());
            curandSetPseudoRandomGeneratorSeed(prng, (unsigned long long) clock());
            CUDA_CHK(cudaGetLastError());
            curandGenerateUniformDouble(prng, d_inMatrix, N);
        }
    }

    { // GM Kernel
        Metric m;
        for (auto i = 0; i < BENCH_TIMES; ++i) {
            auto block_width = 32u;
            auto grid_width = ceilx((double)width / block_width);
            dim3 dimGrid{grid_width, grid_width};
            dim3 dimBlock{block_width, block_width};
            auto t = m.track_begin();
            stencil_kernel_gm<<<dimGrid, dimBlock>>>(d_inMatrix, d_outMatrix1, H*H, N, width);
            CUDA_CHK(cudaGetLastError());
            CUDA_CHK(cudaDeviceSynchronize());
            m.track_end(t);
        }
        printf("Non SHM Kernel: mean: %f ms, stdev: %f, CV: %f\n", m.mean(), m.stdev(), m.cv());
    }

    { // SHM kernel
        Metric m;
        for (auto i = 0; i < BENCH_TIMES; ++i) {
            auto block_width = SHM_BLOCK_WIDTH;
            auto grid_width = ceilx((double)width / block_width);
            dim3 dimGrid{grid_width, grid_width};
            dim3 dimBlock{block_width, block_width};
            auto t = m.track_begin();
            stencil_kernel_shm<<<dimGrid, dimBlock>>>(d_inMatrix, d_outMatrix2, H*H, N, width);
            CUDA_CHK(cudaGetLastError());
            CUDA_CHK(cudaDeviceSynchronize());
            m.track_end(t);
        }
        printf("SHM Kernel: mean: %f ms, stdev: %f, CV: %f\n", m.mean(), m.stdev(), m.cv());
    }

    { // validation
        int h_diff = gpu_compare_arrays<value_type>(d_outMatrix1, d_outMatrix2, N);
        printf("Cmp Result: %d\n", h_diff);
    }

    cudaFree(d_outMatrix2);
    cudaFree(d_outMatrix1);
    cudaFree(d_inMatrix);
}

void ejercicio2()
{
    ejercicio2_impl(4096);
    ejercicio2_impl(8192);
}

#endif // EJERCICIO2_CUH__
