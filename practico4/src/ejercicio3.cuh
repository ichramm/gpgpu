/*!
 * \file ejercicio3.cuh
 * \author Juan Ramirez (juan.ramirez@fing.edu.uy)
 */
#ifndef EJERCICIO3_CUH__
#define EJERCICIO3_CUH__

#include "practico4.h"

// se asume que el tamaño de perm es igual al del bloque
// y que las premutaciones son válidas
__global__ void block_perm(int * data, int *perm, int length){
    int off = blockIdx.x * blockDim.x;
    if (length < off+threadIdx.x) return;
    int perm_data = data[off + perm[threadIdx.x]];
    __syncthreads();
    data[off+threadIdx.x]=perm_data;
}


void ejercicio3() {

}

#endif // EJERCICIO3_CUH__
