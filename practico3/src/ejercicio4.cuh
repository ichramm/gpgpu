#ifndef EJERCICIO_H__
#define EJERCICIO_H__

#include "practico3.h"


// para mejorar:
// padding
// shared mem
// __restrict__ para que cachee
// tiling

// me armo una matriz de memoria compartida
// me copio la parte relevante de memoria global
// lo necesario para todo el block
// opero con la matriz local


__global__ void kernel_ejercicio4_sharedmem(const value_type* __restrict__ input_matrix,
                                            value_type *output_matrix,
                                            value_type h,
                                            unsigned int N,
                                            unsigned int width)
{
    // 0: up, 2: below
    __shared__ value_type shared_mem[3][BLOCK_SIZE];

    auto blockId  = blockIdx.y * gridDim.x + blockIdx.x;
	auto threadId = blockId * blockDim.x + threadIdx.x;

    if (threadId < N) {
        // cell above: (y-1, x)
        shared_mem[0][threadIdx.x] = threadId >= width ? input_matrix[threadId - width] : 0;
        // cell itself: (y, x)
        shared_mem[1][threadIdx.x] = input_matrix[threadId];
        // cell below: (y+1, x)
        shared_mem[2][threadIdx.x] = threadId < N - width ? input_matrix[threadId + width] : 0;
    }

    __syncthreads();

    if (threadId < N) {
        auto x_coord = threadId % width;
        auto value = shared_mem[1][threadIdx.x] * (-4)
                     + (x_coord > 0 ? (threadIdx.x > 0 ? shared_mem[0][threadIdx.x - 1]
                                                       : input_matrix[threadId-1])
                                    : 0) // left
                     + shared_mem[0][threadIdx.x]     // above
                     + (x_coord < width-1 ? (threadIdx.x < BLOCK_SIZE-1 ? shared_mem[1][threadIdx.x + 1]
                                                                        : input_matrix[threadId+1])
                                          : 0) // right
                     + shared_mem[2][threadIdx.x];    // below
        output_matrix[threadId] = value / h;
    }
}

__global__ void kernel_ejercicio4(value_type* input_matrix,
                                  value_type *output_matrix,
                                  value_type h,
                                  unsigned int N,
                                  unsigned int width)
{
    // 2D grid of 1D block
    auto blockId  = blockIdx.y * gridDim.x + blockIdx.x;
	auto threadId = blockId * blockDim.x + threadIdx.x;

    if (threadId < N) {
        auto x_coord = threadId % width;

        auto value = input_matrix[threadId] * (-4)
            + (x_coord > 0 ? input_matrix[threadId-1] : 0) // left: (y, x-1)
            + (x_coord < width-1 ? input_matrix[threadId + 1] : 0) // right: (y, x+1)
            + (threadId >= width ? input_matrix[threadId - width] : 0) // cell above: (y-1, x)
            + (threadId < N - width ? input_matrix[threadId + width] : 0); // cell below: (y+1, x)

        output_matrix[threadId] = value / h;
    }
}

void ejercicio4(bool use_shared_mem = false)
{
    value_type *input_matrix;
    unsigned int N;
    unsigned int width;

    printf("Building input matrix...\n");
    ejercicio1(&input_matrix, &N, &width);
    printf("Input matrix built.\n");

    value_type *output_matrix;
    CUDA_CHK(cudaMalloc(&output_matrix, N * sizeof(value_type)));

    auto dimGrid = dim3(ceilx((double)width/BLOCK_SIZE), width, 1);
    auto dimBlock = dim3(BLOCK_SIZE, 1, 1);

    printf("N:       %6d\n", N);
    printf("Threads: %6d\n", dimGrid.x * dimGrid.y * dimGrid.z * dimBlock.x);

    float elapsedTime;
    CUDA_MEASURE_START();
    if (use_shared_mem) {
        kernel_ejercicio4_sharedmem<<<dimGrid, dimBlock>>>(input_matrix,
                                                           output_matrix,
                                                           0.001,
                                                           N,
                                                           width);
    } else {
        kernel_ejercicio4<<<dimGrid, dimBlock>>>(input_matrix,
                                                 output_matrix,
                                                 0.001,
                                                 N,
                                                 width);
    }
    CUDA_CHK(cudaGetLastError());
    CUDA_CHK(cudaDeviceSynchronize());
    CUDA_MEASURE_STOP(elapsedTime);

    value_type *h_matrix; // = (value_type *)malloc(N * sizeof(value_type));
    CUDA_CHK(cudaMallocHost(&h_matrix, N * sizeof(value_type)));
    CUDA_CHK(cudaMemcpy(h_matrix, output_matrix, N * sizeof(value_type), cudaMemcpyDeviceToHost));

    // TODO: Validate data is correct?

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

    CUDA_CHK(cudaFreeHost(h_matrix));
    CUDA_CHK(cudaFree(input_matrix));
    CUDA_CHK(cudaFree(output_matrix));

    printf("Elapsed time: %f ms\n", elapsedTime);
}

#endif // EJERCICIO_H__
