/*!
 * \file ejercicio4.cuh
 * \author Juan Ramirez (juan.ramirez@fing.edu.uy)
 */
#ifndef EJERCICIO4_CUH__
#define EJERCICIO4_CUH__

#include "practico4.h"

#define TSZ 32

__global__ void sum_col_block(int * data, int length){
    __shared__ int sh_tile[TSZ][TSZ];

    int n = gridDim.x * blockDim.x;
    int idx = blockIdx.x * blockDim.x+threadIdx.x;
    int idy = blockIdx.y * blockDim.y+threadIdx.y;

    sh_tile[threadIdx.y][threadIdx.x] = data[idy*n+idx];
    __syncthreads();
    int col_sum=sh_tile[threadIdx.x][threadIdx.y];

    for (int i=16; i>0; i/=2)
        col_sum+=__shfl_down_sync(0xFFFFFFFF, col_sum, i);

    data[idy*n+idx]=col_sum;
}

// FIXME: aun no cambié nada
__global__ void sum_col_block_2(int * data, int length){
    __shared__ int sh_tile[TSZ][TSZ];

    int n = gridDim.x * blockDim.x;
    int idx = blockIdx.x * blockDim.x+threadIdx.x;
    int idy = blockIdx.y * blockDim.y+threadIdx.y;

    sh_tile[threadIdx.y][threadIdx.x] = data[idy*n+idx];
    __syncthreads();
    int col_sum=sh_tile[threadIdx.x][threadIdx.y];

    for (int i=16; i>0; i/=2)
        col_sum+=__shfl_down_sync(0xFFFFFFFF, col_sum, i);

    data[idy*n+idx]=col_sum;
}

__global__ void cmp_kernel(int * data1, int *data2, int length, int *ndiff) {
    int n = gridDim.x * blockDim.x;
    int idx = blockIdx.x * blockDim.x+threadIdx.x;
    int idy = blockIdx.y * blockDim.y+threadIdx.y;

    if (data1[idy*n+idx] != data2[idy*n+idx]) {
        atomicAdd(ndiff, 1);
    }
}

void ejercicio4() {
    constexpr int width = 8192;
    constexpr int block_width = 32;
    constexpr int grid_witdh = width / block_width;
    constexpr int shared_mem_size = sizeof(int) * block_width * block_width;
    constexpr int N = width * width;

    int *d_data1, *d_data2;
    CUDA_CHK(cudaMalloc(&d_data1, sizeof(int) * N));
    CUDA_CHK(cudaMalloc(&d_data2, sizeof(int) * N));

    { // setup
        curandGenerator_t prng;
        curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_XORWOW);
        curandSetPseudoRandomGeneratorSeed(prng, (unsigned long long) clock());
        curandGenerate(prng, (unsigned int *)(d_data1), N);
        cudaMemcpy(d_data2, d_data1, sizeof(int) * N, cudaMemcpyDeviceToDevice);
    }

    { // provided kernel
        Metric m;
        dim3 dimGrid(grid_witdh, grid_witdh);
        dim3 dimBlock(block_width, block_width);
        auto t = m.track_begin();
        sum_col_block<<<dimGrid, dimBlock, shared_mem_size>>>(d_data1, N);
        m.track_end(t);
        printf("Kernel 1 - Time: %f\n", m.total());
    }

    { // improved kernel
        Metric m;
        dim3 dimGrid(grid_witdh, grid_witdh);
        dim3 dimBlock(block_width, block_width);
        auto t = m.track_begin();
        sum_col_block_2<<<dimGrid, dimBlock, shared_mem_size>>>(d_data2, N);
        m.track_end(t);
        printf("Kernel 2 - Time: %f\n", m.total());
    }

    { // compare
        int h_diff;
        int *d_diff;
        cudaMalloc(&d_diff, sizeof(int));
        dim3 dimGrid(grid_witdh, grid_witdh);
        dim3 dimBlock(block_width, block_width);
        cmp_kernel<<<dimGrid, dimBlock>>>(d_data1, d_data2, N, d_diff);
        cudaMemcpy(&h_diff, d_diff, sizeof(int), cudaMemcpyDeviceToHost);
        printf("Cmp Result: %d\n", h_diff);
        cudaFree(d_diff);
    }

    cudaFree(d_data2);
    cudaFree(d_data1);
}

#endif // EJERCICIO4_CUH__
