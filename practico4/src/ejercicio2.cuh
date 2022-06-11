/*!
 * \file ejercicio2.cuh
 * \author Juan Ramirez (juan.ramirez@fing.edu.uy)
 */
#ifndef EJERCICIO2_CUH__
#define EJERCICIO2_CUH__

#include "practico4.h"


/*!
 * Kernel del practico 3 adaptado a este stencil
 */
__global__ void stencil_kernel_gm(const value_type * __restrict__ in,
                                  value_type *out,
                                  value_type hsq,
                                  uint32_t N,
                                  uint32_t width)
{
    // 2D grid of 1D block
    auto blockId  = blockIdx.y * gridDim.x + blockIdx.x;
    auto threadId = blockId * blockDim.x + threadIdx.x;

    if (threadId < N) {
        auto x_coord = threadId % width;

        auto value = in[threadId] * (-8)
            // Din[x+1,y] + Din[x+2,y] + Din[x-1,y] + Din[x-2,y] +
            + (x_coord < width-1 ? in[threadId + 1] : 0)                // right: (y, x+1)
            + (x_coord < width-2 ? in[threadId + 2] : 0)                // right: (y, x+2)
            + (x_coord > 1 ? in[threadId-2] : 0)                        // left : (y, x-2)
            + (x_coord > 0 ? in[threadId-1] : 0)                        // left : (y, x-1)
            + (threadId < N - width ? in[threadId + width] : 0)         // below: (y+1, x)
            + (threadId < N - (2*width) ? in[threadId + (2*width)] : 0) // below: (y+2, x)
            + (threadId >= width ? in[threadId - width] : 0)            // above: (y-1, x)
            + (threadId >= (2*width) ? in[threadId - (2*width)] : 0);   // above: (y-2, x)

        out[threadId] = value / hsq;
    }
}

#ifndef BLOCK_DIM
#define BLOCK_DIM 16u
#endif


/*!
 * Este Kernel utiliza un tile de ancho BLOCK_SIZE+2.
 * Los elementos del extremo se leen directamente de memoria global dado
 * que solo son leidos una vez.
 */
__global__ void stencil_kernel_shm_bitfaster(const value_type * __restrict__ in,
                                   value_type *out,
                                   value_type hsq,
                                   uint32_t N,
                                   uint32_t width)
{
    __shared__ value_type tile[BLOCK_DIM+2][BLOCK_DIM+3];

    int blockId = blockIdx.x + blockIdx.y * gridDim.x;
    int threadId = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;

    // TODO: No usar division y modulo

//     int x = blockIdx.x*blockDim.x + threadIdx.x;
//     int y = blockIdx.y*blockDim.y + threadIdx.y;
//     if (x < dimX && y < dimY)
//         mem[y*dimX+x] = sin( h*(x-y) );

    auto g_x = blockIdx.x*blockDim.x + threadIdx.x; // threadId % width;
    auto g_y = blockIdx.y*blockDim.y + threadIdx.y; // threadId / width;

    // one-element halo
    auto shm_x = threadIdx.x + 1;
    auto shm_y = threadIdx.y + 1;

    auto g_idx = g_y*width + g_x;

    if (g_x < width && g_y < width) {
        tile[shm_y][shm_x] = in[g_idx];

        if (threadIdx.x == 0) {
            tile[shm_y][shm_x-1] = g_x > 0 ? in[g_idx-1]: 0;
        }
        if (threadIdx.x == blockDim.x - 1) {
            tile[shm_y][shm_x+1] = (g_x < width - 1) ? in[g_idx+1]: 0;
        }
        if (threadIdx.y == 0) {
            tile[shm_y-1][shm_x] = g_y > 0 ? in[g_idx-width]: 0;
        }
        if (threadIdx.y == blockDim.y - 1) {
            tile[shm_y+1][shm_x] = (g_idx < N - width) ? in[g_idx+width] : 0;
        }
    }

    __syncthreads();

    if (g_x < width && g_y < width) {
#if 0
        auto value = tile[shm_y][shm_x] * (-8);
        value += tile[shm_y][shm_x+1];
        value += g_idx < N - 2 ? in[g_idx+2] : 0; //tile[shm_y][shm_x+2] +
        value += tile[shm_y][shm_x-1];
        value += g_idx > 2 ? in[g_idx-2] : 0; // tile[shm_y][shm_x-2] +
        value += tile[shm_y+1][shm_x];
        value += (g_idx < (N - 2*width)) ? in[g_idx+(2*width)] : 0; // tile[shm_y+2][shm_x] +
        value += tile[shm_y-1][shm_x];
        value += (g_idx > (2*width)) ? in[g_idx-(2*width)] : 0; // tile[shm_y-2][shm_x];
#else
        auto value = tile[shm_y][shm_x] * (-8) +
            tile[shm_y][shm_x+1]+
            (g_idx < N - 2 ? in[g_idx+2] : 0) + //tile[shm_y][shm_x+2] +
            tile[shm_y][shm_x-1] +
            (g_idx > 2 ? in[g_idx-2] : 0) + // tile[shm_y][shm_x-2] +
            tile[shm_y+1][shm_x] +
            (g_idx < N - 2*width ? in[g_idx+(2*width)] : 0) + // tile[shm_y+2][shm_x] +
            tile[shm_y-1][shm_x] +
            (g_idx > 2*width ? in[g_idx-(2*width)] : 0); // tile[shm_y-2][shm_x];
#endif

        out[g_idx] = value / hsq;
    }
}


/*!
 * Este Kernel utiliza un tile de ancho BLOCK_SIZE+4.
 * A diferencia del kernel anterior, todos los elementos necesarios se copian de memoria
 * global a la memoria compartid.
 * DOES NOT CRASH ANYMORE
 */
__global__ void stencil_kernel_shm_crash(const value_type * __restrict__ in,
                                   value_type *out,
                                   value_type hsq,
                                   uint32_t N,
                                   uint32_t width)
{
    __shared__ value_type tile[BLOCK_DIM+4][BLOCK_DIM+5];

    int blockId = blockIdx.x + blockIdx.y * gridDim.x;
    int threadId = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;

#if 0
    auto g_x = threadId % width;
    auto g_y = threadId / width;
    auto shm_x = threadIdx.x + 4;
    auto shm_y = threadIdx.y + 4;
#endif

    auto g_x = blockIdx.x*blockDim.x + threadIdx.x; // threadId % width;
    auto g_y = blockIdx.y*blockDim.y + threadIdx.y; // threadId / width;

    // one-element halo
    auto shm_x = threadIdx.x + 2;
    auto shm_y = threadIdx.y + 2;

    auto g_idx = g_y*width + g_x;

    if (g_x < width && g_y < width) {

        tile[shm_y][shm_x] = in[g_idx];

        if (threadIdx.x < 2) {
            tile[shm_y][shm_x-2] = g_x > 1 ? in[g_idx-2] : 0;
        }
        if (threadIdx.x >= blockDim.x - 2) {
            tile[shm_y][shm_x+2] = g_x < width - 2 ? in[g_idx+2]: 0;
        }
        if (threadIdx.y < 2) {
            tile[shm_y-2][shm_x] = g_y > 1 ? in[g_idx-(2*width)]: 0;
        }
        if (threadIdx.y >= blockDim.y - 2) {
            tile[shm_y+2][shm_x] = g_idx < N - (2 * width) ? in[g_idx+(2*width)]: 0;
        }
    }

    __syncthreads();

    if (g_x < width && g_y < width) {
        auto value = tile[shm_y][shm_x] * (-8) +
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


//  > Width: 8192
// Non SHM Kernel: mean: 18.688149 ms, stdev: 0.059954, CV: 0.003208
// SHM Kernel: mean: 19.016980 ms, stdev: 0.009346, CV: 0.000491
__global__ void stencil_kernel_shm(const value_type * __restrict__ in,
                                   value_type *out,
                                   value_type hsq,
                                   uint32_t N,
                                   uint32_t width)
{
    // size: (block-size+4)*(block-size+4)
    __shared__ value_type tile[(BLOCK_DIM+4)*(BLOCK_DIM+4)];

    auto blockThreadId = threadIdx.y * blockDim.x + threadIdx.x;
    auto offset = blockDim.x * 2 + 2;
    auto blockThreadIdx = blockThreadId + offset;
    //auto blockThreadIdx = blockThreadId;

    auto g_x = blockIdx.x*blockDim.x + threadIdx.x; // threadId % width;
    auto g_y = blockIdx.y*blockDim.y + threadIdx.y; // threadId / width;
    auto threadId = g_y*width + g_x;

    if (threadId >= N) {
        return;
    }

    //if (threadId < N) {
        tile[blockThreadIdx] = in[threadId];

        // threads in the first row load values above
        if (threadIdx.y < 2) {
            tile[blockThreadIdx - 2 * blockDim.x] = threadId > 2*width ? in[threadId - 2*width] : 0;
        }

        // threads in the last row load values below
        if (threadIdx.y > blockDim.y - 2) {
            tile[blockThreadIdx + blockDim.x * 2] = threadId < N - width * 2 ? in[threadId + width * 2] : 0;
        }
    //}

    __syncthreads();

    //if (threadId < N) {
        auto value = tile[blockThreadIdx] * (-8) +
                     tile[blockThreadIdx + 1] +
                     tile[blockThreadIdx + 2] +
                     tile[blockThreadIdx - 1] +
                     tile[blockThreadIdx - 2] +
                     tile[blockThreadIdx + blockDim.x] +
                     tile[blockThreadIdx + 2 * blockDim.x] +
                     tile[blockThreadIdx - blockDim.x] +
                     tile[blockThreadIdx - 2 * blockDim.x];

        out[threadId] = value / hsq;
   // }
}



void ejercicio2_impl(int32_t width) {
    printf(" > Width: %u\n", width);
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

    { // kernel 1
        Metric m;
        for (auto i = 0; i < BENCH_TIMES; ++i) {
            auto dimGrid = dim3(ceilx((double)width/BLOCK_SIZE), width, 1);
            auto dimBlock = dim3(BLOCK_SIZE, 1, 1);
            auto t = m.track_begin();
            stencil_kernel_gm<<<dimGrid, dimBlock>>>(d_inMatrix, d_outMatrix1, H*H, N, width);
            CUDA_CHK(cudaGetLastError());
            CUDA_CHK(cudaDeviceSynchronize());
            m.track_end(t);
        }
        printf("Non SHM Kernel: mean: %f ms, stdev: %f, CV: %f\n", m.mean(), m.stdev(), m.cv());
    }


    { // kernel 2
        Metric m;
        for (auto i = 0; i < BENCH_TIMES; ++i) {
            auto block_width = BLOCK_DIM;
            auto grid_width = ceilx((double)width / block_width);
            auto tile_width = block_width + 4;
            auto dimGrid = dim3(grid_width, grid_width, 1);
            auto dimBlock = dim3(block_width, block_width, 1);

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
    ejercicio2_impl(1024);
    ejercicio2_impl(2048);
    ejercicio2_impl(4096);
    ejercicio2_impl(8192);
}

#endif // EJERCICIO2_CUH__
