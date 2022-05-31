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

void ejercicio4() {

}

#endif // EJERCICIO4_CUH__
