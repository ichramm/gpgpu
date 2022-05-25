#ifndef EJERCICIO3_CUH__
#define EJERCICIO3_CUH__

#include "practico3.h"

__global__ void kernel_ejercicio3a(const value_type* __restrict__ from_cube,
                                   value_type *to_matrix,
                                   unsigned int N,
                                   unsigned int width)
{
    auto blockId  = blockIdx.y * gridDim.x + blockIdx.x;
	auto threadId = blockId * blockDim.x + threadIdx.x;

    if (threadId < N) {
        // matrix[y, x] = sum (z from -pi to pi) f(x, y, z)
        auto y_coord = threadId / width;
        auto x_coord = threadId % width;

        value_type acc = 0;
        for (auto z_coord = 0u; z_coord < width; ++z_coord) {
            acc += from_cube[COORD3IDX(x_coord, y_coord, z_coord, width, width)];
        }

        to_matrix[threadId] = acc;
    }
}

__global__ void kernel_ejercicio3b(const value_type* __restrict__ from_cube,
                                   value_type *to_matrix,
                                   unsigned int N,
                                   unsigned int width)
{
    auto blockId  = blockIdx.y * gridDim.x + blockIdx.x;
	auto threadId = blockId * blockDim.x + threadIdx.x;

    if (threadId < N) {
        // matrix[z, x] = sum (y from -pi to pi) f(x, y, z)
        auto z_coord = threadId / width;
        auto x_coord = threadId % width;

        value_type acc = 0;
        for (auto y_coord = 0u; y_coord < width; ++y_coord) {
            acc += from_cube[COORD3IDX(x_coord, y_coord, z_coord, width, width)];
        }

        to_matrix[threadId] = acc;
    }
}

__global__ void kernel_ejercicio3c(const value_type* __restrict__ from_cube,
                                   value_type *to_matrix,
                                   unsigned int N,
                                   unsigned int width)
{
    auto blockId  = blockIdx.y * gridDim.x + blockIdx.x;
	auto threadId = blockId * blockDim.x + threadIdx.x;

    if (threadId < N) {
        // matrix[z, y] = sum (x from -pi to pi) f(x, y, z)
        auto z_coord = threadId / width;
        auto y_coord = threadId % width;

        value_type acc = 0;
        for (auto x_coord = 0u; x_coord < width; ++x_coord) {
            acc += from_cube[COORD3IDX(x_coord, y_coord, z_coord, width, width)];
        }

        to_matrix[threadId] = acc;
    }
}

static void ejercicio3(char part) {
    value_type *d_cube;
    unsigned int n_cube;
    unsigned int width;

    printf("Building cube...\n");
    ejercicio2(&d_cube, &n_cube, &width);
    printf("Cube built.\n");

    auto N = width * width;

    value_type *d_matrix;
    CUDA_CHK(cudaMalloc(&d_matrix, N * sizeof(value_type)));

    // 2D grid of 1D blocks
    auto dimGrid = dim3(ceilx((double)width/BLOCK_SIZE), width, 1);
    auto dimBlock = dim3(BLOCK_SIZE, 1, 1);

    printf("N:       %6d\n", N);
    printf("Threads: %6d\n", dimGrid.x * dimGrid.y * dimGrid.z * dimBlock.x);

    float elapsedTime;
    CUDA_MEASURE_START();
    switch (part) {
        case 'a':
            kernel_ejercicio3a<<<dimGrid, dimBlock>>>(d_cube, d_matrix, N, width);
            break;
        case 'b':
            kernel_ejercicio3b<<<dimGrid, dimBlock>>>(d_cube, d_matrix, N, width);
            break;
        case 'c':
            kernel_ejercicio3c<<<dimGrid, dimBlock>>>(d_cube, d_matrix, N, width);
            break;
    }
    CUDA_CHK(cudaGetLastError());
    CUDA_CHK(cudaDeviceSynchronize());
    CUDA_MEASURE_STOP(elapsedTime);

    value_type *h_matrix; // = (value_type *)malloc(N * sizeof(value_type));
    CUDA_CHK(cudaMallocHost(&h_matrix, N * sizeof(value_type)));
    CUDA_CHK(cudaMemcpy(h_matrix, d_matrix, N * sizeof(value_type), cudaMemcpyDeviceToHost));

    // TODO: Validate data is correct?

    printf("Some results:\n");
    for (auto y : fibonacci_numbers) {
        if (y > width) break;
        if (!(y & 1)) continue; // skip some
        for (auto x : fibonacci_numbers) {
            if (x > width) break;
            if (x & 1) continue; // skip some
            printf("   (%3d, %d): %s", y, x, val2string(h_matrix[COORD2IDX(x, y, width)], "% 5.3f", 9).c_str());
        }
        printf("\n");
    }

    cudaFreeHost(h_matrix);
    cudaFree(d_matrix);
    cudaFree(d_cube);

    printf("Kernel elapsed time: %f ms\n", elapsedTime);
}

void ejercicio3a() {
    ejercicio3('a');
}

void ejercicio3b() {
    ejercicio3('b');
}

void ejercicio3c() {
    ejercicio3('c');
}

#endif // EJERCICIO3_CUH__
