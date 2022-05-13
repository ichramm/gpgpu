#ifndef EJERCICIO3_CUH__
#define EJERCICIO3_CUH__

#include "practico3.h"

__global__ void kernel_ejercicio3a(value_type *from_cube,
                                   value_type *to_matrix,
                                   unsigned int N,
                                   unsigned int width,
                                   unsigned int mem_xdim,  // same as width if there's no padding
                                   unsigned int mem_ydim)
{
    auto blockId  = blockIdx.y * gridDim.x + blockIdx.x;
	auto threadId = blockId * blockDim.x + threadIdx.x;

    if (threadId < N) {
        // matrix[y, x] = sum (z from -pi to pi) f(x, y, z)
        auto x_coord = threadId % mem_xdim;
        if (x_coord < width) { // always true if there's no padding
            auto y_coord = threadId / mem_xdim;

            value_type acc = 0;
            for (auto z_coord = 0u; z_coord < width; ++z_coord) {
                acc += from_cube[COORD3IDX(x_coord, y_coord, z_coord, mem_xdim, mem_ydim)];
            }

            to_matrix[threadId] = acc;
        }
    }
}

__global__ void kernel_ejercicio3b(value_type *from_cube,
                                   value_type *to_matrix,
                                   unsigned int N,
                                   unsigned int width,
                                   unsigned int mem_xdim,  // same as width if there's no padding
                                   unsigned int mem_ydim)
{
    auto blockId  = blockIdx.y * gridDim.x + blockIdx.x;
	auto threadId = blockId * blockDim.x + threadIdx.x;

    if (threadId < N) {
        // matrix[z, x] = sum (y from -pi to pi) f(x, y, z)
        auto x_coord = threadId % mem_xdim;
        if (x_coord < width) { // always true if there's no padding
            auto z_coord = threadId / mem_xdim;

            value_type acc = 0;
            for (auto y_coord = 0u; y_coord < width; ++y_coord) {
                acc += from_cube[COORD3IDX(x_coord, y_coord, z_coord, mem_xdim, mem_ydim)];
            }

            to_matrix[threadId] = acc;
        }
    }
}

__global__ void kernel_ejercicio3c(value_type *from_cube,
                                   value_type *to_matrix,
                                   unsigned int N,
                                   unsigned int width,
                                   unsigned int mem_xdim,  // same as width if there's no padding
                                   unsigned int mem_ydim)
{
    auto blockId  = blockIdx.y * gridDim.x + blockIdx.x;
	auto threadId = blockId * blockDim.x + threadIdx.x;

    if (threadId < N) {
        // matrix[z, y] = sum (x from -pi to pi) f(x, y, z)
        auto y_coord = threadId % mem_xdim;
        if (y_coord < width) { // padding at row-level
            auto z_coord = threadId / mem_xdim;

            value_type acc = 0;
            for (auto x_coord = 0u; x_coord < width; ++x_coord) {
                acc += from_cube[COORD3IDX(x_coord, y_coord, z_coord, mem_xdim, mem_ydim)];
            }

            to_matrix[threadId] = acc;
        }
    }
}


__global__ void kernel_ejercicio3cs(value_type *from_cube,
                                    value_type *to_matrix,
                                    unsigned int N,
                                    unsigned int width,
                                    unsigned int mem_xdim,  // same as width if there's no padding
                                    unsigned int mem_ydim)
{
    __shared__ value_type shared_mem[BLOCK_SIZE];

    auto blockId  = blockIdx.y * gridDim.x + blockIdx.x;
	auto threadId = blockId * blockDim.x + threadIdx.x;

    if (threadId < N) {
        // matrix[z, y] = sum (x from -pi to pi) f(x, y, z)
        auto y_coord = threadId % mem_xdim;
        if (y_coord < width) { // padding at row-level
            auto z_coord = threadId / mem_xdim;

            value_type acc = 0;
            for (auto x_coord = 0u; x_coord < width; ++x_coord) {
                acc += from_cube[COORD3IDX(x_coord, y_coord, z_coord, mem_xdim, mem_ydim)];
            }

            shared_mem[threadIdx.x] = acc;
        }
    }

    __syncthreads();

    if (threadIdx.x == 0 && threadId < N) {
        memcpy(to_matrix + threadId,
               shared_mem,
               min(sizeof(shared_mem), sizeof(value_type) * (N - threadId)));
    }
}

static void ejercicio3(char part) {
    value_type *d_cube;
    unsigned int n_cube;
    unsigned int width;
    unsigned int mem_xdim;
    unsigned int mem_ydim;
    unsigned int mem_zdim;

    printf("Building cube...\n");
    ejercicio2(&d_cube, &n_cube, &width, &mem_xdim, &mem_ydim, &mem_zdim);
    printf("Cube built.\n");

    auto N = mem_ydim * mem_xdim;

    value_type *d_matrix;
    cudaMalloc(&d_matrix, N * sizeof(value_type));

    // 2D grid of 1D blocks
    auto dimGrid = dim3(ceilx((double)mem_xdim/BLOCK_SIZE), mem_ydim, 1);
    auto dimBlock = dim3(BLOCK_SIZE, 1, 1);

    printf("N:       %6d\n", N);
    printf("Threads: %6d\n", dimGrid.x * dimGrid.y * dimGrid.z * dimBlock.x);

    float elapsedTime;
    CUDA_MEASURE_START();
    switch (part) {
        case 'a':
            kernel_ejercicio3a<<<dimGrid, dimBlock>>>(d_cube, d_matrix, N, width, mem_xdim, mem_ydim);
            break;
        case 'b':
            kernel_ejercicio3b<<<dimGrid, dimBlock>>>(d_cube, d_matrix, N, width, mem_xdim, mem_ydim);
            break;
        case 'c':
            kernel_ejercicio3c<<<dimGrid, dimBlock>>>(d_cube, d_matrix, N, width, mem_xdim, mem_ydim);
            break;
        case 'C':
            kernel_ejercicio3cs<<<dimGrid, dimBlock>>>(d_cube, d_matrix, N, width, mem_xdim, mem_ydim);
            break;
    }
    CUDA_MEASURE_STOP(elapsedTime);
    printf("Elapsed time: %f ms\n", elapsedTime);

    value_type *h_matrix; // = (value_type *)malloc(N * sizeof(value_type));
    cudaMallocHost(&h_matrix, N * sizeof(value_type));
    cudaMemcpy(h_matrix, d_matrix, N * sizeof(value_type), cudaMemcpyDeviceToHost);

    // TODO: Validate data is correct?

    printf("Some results:\n");
    for (auto y : fibonacci_numbers) {
        if (y > width) break;
        if (!(y & 1)) continue; // skip some
        for (auto x : fibonacci_numbers) {
            if (x > width) break;
            if (x & 1) continue; // skip some
            auto valx = COORDVALX(x, 0.01);
            auto valy = COORDVALY(y, 0.01);
            printf("\t(%3d, %d): %s", y, x, val2string(h_matrix[COORD2IDX(x, y, mem_xdim)], 8).c_str());
        }
        printf("\n");
    }

    cudaFree(h_matrix);
    cudaFree(d_matrix);
    cudaFree(d_cube);
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

void ejercicio3cs() {
    ejercicio3('C');
}

#endif // EJERCICIO3_CUH__
