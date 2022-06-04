/*!
 * \file ejercicio2.cuh
 * \author Juan Ramirez (juan.ramirez@fing.edu.uy)
 */
#ifndef EJERCICIO2_CUH__
#define EJERCICIO2_CUH__

#include "practico4.h"

/*
a) Extienda el ejercicio 4 del práctico 3 para que calcule el siguiente stencil:
    Dout[x,y] = ( (-8)Din[x,y] +
                   Din[x+1,y] + Din[x+2,y] + Din[x-1,y] + Din[x-2,y] +
                   Din[x,y+1] + Din[x,y+2] + Din[x,y-1] + Din[x,y-2] ) / h^2

b) Utilice la memoria compartida para reutilizar los datos cargados por hilos vecinos. La configuración de
la grilla de hilos debe asociar un hilo a cada elemento de la grilla (matrices din y dout). La región de
memoria compartida a usar debe contemplar todos los elementos accedidos por cada bloque. La carga de
la memoria compartida debe realizarse con el mayor paralelismo posible.

c) Compare el desempeño de los kernels correspondientes a las partes a) y b) para grillas de tamaño 4096 2
y 81922. No es necesario reservar memoria en CPU ni transferir las matrices. Pueden generarse en GPU
mediante un kernel reservando previamente la memoria necesaria con cudaMalloc.
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

#define TILE_SIZE (BLOCK_DIM-4)


// voy a tratar de poner un halo solo
__global__ void stencil_kernel_shm_bitfaster(const value_type * __restrict__ in,
                                   value_type *out,
                                   value_type hsq,
                                   uint32_t N,
                                   uint32_t width)
{
    __shared__ value_type tile[BLOCK_DIM+2][BLOCK_DIM+2];

    int blockId = blockIdx.x + blockIdx.y * gridDim.x;
    int threadId = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;

    auto g_x = threadId % width;
    auto g_y = threadId / width;
    auto s_x = threadIdx.x + 2;
    auto s_y = threadIdx.y + 2;

    if (threadId < N) {
        tile[s_y][s_x] = in[threadId];

        if (threadIdx.x == 0) {
            tile[s_y][s_x-1] = g_x > 0 ? in[threadId-1]: 0;
        }
        if (threadIdx.x == blockDim.x - 1) {
            tile[s_y][s_x+1] = g_x < width - 1 ? in[threadId+1]: 0;
        }
        if (threadIdx.y == 0) {
            tile[s_y-1][s_x] = g_y > 0 ? in[threadId-width]: 0;
        }
        if (threadIdx.y == blockDim.y - 1) {
            tile[s_y+1][s_x] = g_y < N - width ? in[threadId+width]: 0;
        }
    }

    __syncthreads();

    if (threadId < N) {
        auto value = tile[s_y][s_x] * (-8) +
                        tile[s_y][s_x+1] +
                        threadId < N - 2 ? in[threadId+2] : 0 + //tile[s_y][s_x+2] +
                        tile[s_y][s_x-1] +
                        threadId > 2 ? in[threadId-2] : 0 + // tile[s_y][s_x-2] +
                        tile[s_y+1][s_x] +
                        threadId < N - (2*width) ? in[threadId+(2*width)] : 0 + // tile[s_y+2][s_x] +
                        tile[s_y-1][s_x] +
                        threadId > (2*width) ? in[threadId-(2*width)] : 0;// tile[s_y-2][s_x];

        out[threadId] = value / hsq;
    }
}



__global__ void stencil_kernel_shm_crash(const value_type * __restrict__ in,
                                   value_type *out,
                                   value_type hsq,
                                   uint32_t N,
                                   uint32_t width)
{
    __shared__ value_type tile[BLOCK_DIM+4][BLOCK_DIM+4];

    int blockId = blockIdx.x + blockIdx.y * gridDim.x;
    int threadId = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;

    auto g_x = threadId % width;
    auto g_y = threadId / width;
    auto s_x = threadIdx.x + 4;
    auto s_y = threadIdx.y + 4;

    if (threadId < N) {

        tile[s_y][s_x] = in[threadId];

        if (threadIdx.x < 2) {
            tile[s_y][s_x-2] = g_x > 1 ? in[threadId-2]: 0;
        }
        if (threadIdx.x > blockDim.x - 2) {
            tile[s_y][s_x+2] = g_x < width - 2 ? in[threadId+2]: 0;
        }
        if (threadIdx.y < 2) {
            tile[s_y-2][s_x] = g_y > 1 ? in[threadId-(2*width)]: 0;
        }
        if (threadIdx.y > blockDim.y - 2) {
            tile[s_y+2][s_x] = g_y < N - (2 * width) ? in[threadId+(2*width)]: 0;
        }
    }

    __syncthreads();

    if (threadId < N) {
        auto value = tile[s_y][s_x] * (-8) +
                        tile[s_y][s_x+1] +
                        tile[s_y][s_x+2] +
                        tile[s_y][s_x-1] +
                        tile[s_y][s_x-2] +
                        tile[s_y+1][s_x] +
                        tile[s_y+2][s_x] +
                        tile[s_y-1][s_x] +
                        tile[s_y-2][s_x];

        out[threadId] = value / hsq;
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

    // 2D grid of 2D blocks
    int blockId = blockIdx.x + blockIdx.y * gridDim.x;
	int threadId = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;

    if (threadId >= N) {
        return;
    }

    if (threadId < N) {
        tile[blockThreadIdx] = in[threadId];

        // threads in the first row load values above
        if (threadIdx.y < 2) {
            tile[blockThreadIdx - 2 * blockDim.x] = threadId > 2*width ? in[threadId - 2*width] : 0;
        }

        // threads in the last row load values below
        if (threadIdx.y > blockDim.y - 2) {
            tile[blockThreadIdx + blockDim.x * 2] = threadId < N - width * 2 ? in[threadId + width * 2] : 0;
        }
    }

    __syncthreads();

    if (threadId < N) {
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
    }
}

// TILE_SIZE = BLOCK_SIZE - 4
__global__ void stencil_kernel_shm_slower(const value_type * __restrict__ in,
                                   value_type *out,
                                   value_type hsq,
                                   uint32_t N,
                                   uint32_t width) {
    // tile with halo, one bucket per thread
    __shared__ value_type tile[BLOCK_DIM*BLOCK_DIM];
    //extern __shared__ value_type tile[];

    auto s_idx = threadIdx.y * blockDim.x + threadIdx.x;

    auto tx = threadIdx.x + blockIdx.x*TILE_SIZE - 2;
    auto ty = threadIdx.y + blockIdx.y*TILE_SIZE - 2;

    if (tx < 2 || ty < 2 || tx > width - 2 || ty > width - 2) {
        tile[s_idx] = 0;
    } else {
        tile[s_idx] = in[ty * width + tx];
    }

    __syncthreads();

    if (not (threadIdx.x < 2 || threadIdx.y < 2 || threadIdx.x > blockDim.x - 2 || threadIdx.y > blockDim.y - 2)) {
        auto value = tile[s_idx] * (-8) +
                     tile[s_idx + 1] +
                     tile[s_idx + 2] +
                     tile[s_idx - 1] +
                     tile[s_idx - 2] +
                     tile[s_idx + blockDim.x] +
                     tile[s_idx + 2 * blockDim.x] +
                     tile[s_idx - blockDim.x] +
                     tile[s_idx - 2 * blockDim.x];
        out[ty * width + tx] = value / hsq;
    }
}




void ejercicio2_impl(int32_t width) {
    printf(" > Width: %u\n", width);
    constexpr value_type H = 0.001;
    value_type *d_inMatrix, *d_outMatrix;
    int32_t N = width*width;

    { // setup
        CUDA_CHK(cudaMalloc(&d_inMatrix, sizeof(value_type) * N));
        CUDA_CHK(cudaMalloc(&d_outMatrix, sizeof(value_type) * N));

        {
            curandGenerator_t prng;
            curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_XORWOW);
            CUDA_CHK(cudaGetLastError());
            curandSetPseudoRandomGeneratorSeed(prng, (unsigned long long) clock());
            CUDA_CHK(cudaGetLastError());
            curandGenerateUniformDouble(prng, d_inMatrix, N);
        }
    }

    {
        Metric m;
        for (auto i = 0; i < BENCH_TIMES; ++i) {
            auto dimGrid = dim3(ceilx((double)width/BLOCK_SIZE), width, 1);
            auto dimBlock = dim3(BLOCK_SIZE, 1, 1);
            auto t = m.track_begin();
            stencil_kernel_gm<<<dimGrid, dimBlock>>>(d_inMatrix, d_outMatrix, H*H, N, width);
            CUDA_CHK(cudaGetLastError());
            CUDA_CHK(cudaDeviceSynchronize());
            m.track_end(t);
        }

        printf("Non SHM Kernel: mean: %f ms, stdev: %f, CV: %f\n", m.mean(), m.stdev(), m.cv());
    }

#if 1
    {
        Metric m;
        for (auto i = 0; i < BENCH_TIMES; ++i) {

#if 1
            auto block_width = BLOCK_DIM;
            auto grid_width = ceilx((double)width / (block_width));
            auto tile_width = block_width + 4;
            auto shared_mem_size = sizeof(value_type) * tile_width * tile_width;
#else
            auto block_width = BLOCK_DIM;
            auto grid_width = ceilx((double)width / TILE_SIZE); // tantos blques como tiles
            auto shared_mem_size = sizeof(value_type) * (BLOCK_DIM+4) * (BLOCK_DIM+4);
#endif

            auto dimGrid = dim3(grid_width, grid_width, 1);
            auto dimBlock = dim3(block_width, block_width, 1);

            auto t = m.track_begin();
            stencil_kernel_shm<<<dimGrid, dimBlock, shared_mem_size>>>(d_inMatrix, d_outMatrix, H*H, N, width);
            CUDA_CHK(cudaGetLastError());
            CUDA_CHK(cudaDeviceSynchronize());
            m.track_end(t);
        }

        printf("SHM Kernel: mean: %f ms, stdev: %f, CV: %f\n", m.mean(), m.stdev(), m.cv());
    }
#endif

    cudaFree(d_outMatrix);
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
