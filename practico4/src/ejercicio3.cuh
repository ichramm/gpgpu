/*!
 * \file ejercicio3.cuh
 * \author Juan Ramirez (juan.ramirez@fing.edu.uy)
 */
#ifndef EJERCICIO3_CUH__
#define EJERCICIO3_CUH__

#include "practico4.h"

#include <numeric>
#include <array>
#include <algorithm>

// se asume que el tamaño de perm es igual al del bloque
// y que las premutaciones son válidas
__global__ void block_perm(int * data, int *perm, int length){
    int off = blockIdx.x * blockDim.x;
    if (length < off+threadIdx.x) return;
    int perm_data = data[off + perm[threadIdx.x]];
    __syncthreads();
    data[off+threadIdx.x]=perm_data;
}

/*!
 * Como la permutación se realiza dentro del bloque, copio el segmento
 * de memoria correspondiente a la memoria compartida.
 *
 * Como el tamaño de `perm` es acotado (igual al tamaño del bloque), se
 * utiliza el hint const __restrict__ para indicarle al compilador que
 * el valor puede ser almacenado en el cache.
 */
__global__ void block_perm_shm(int * data, const int * __restrict__ perm, int length) {
    extern __shared__ int shm_data[];

    int id = blockIdx.x * blockDim.x + +threadIdx.x;
    if (id >= length) return;

    // permutation is guaranteed to be inside the segment, so
    // step 1: coalesced copy from global memory to shared memory
    // step 2: read from shared memory
    shm_data[threadIdx.x] = data[id];

    __syncthreads();

    // this will bank-conflict
    data[id] = shm_data[perm[threadIdx.x]];
}


template <int LenLog2> void ejercicio3_impl()
{
    constexpr unsigned int length = (1 << LenLog2);
    constexpr unsigned int block_size = 256;
    constexpr unsigned int grid_size = length / block_size;
    constexpr unsigned int shared_mem_size = sizeof(int) * block_size;

    std::srand(35141);

    std::array<int, block_size> h_perm;
    int *h_data = new int[length];
    int *h_res1 = new int[length];
    int *h_res2 = new int[length];;

    std::generate(h_data, h_data+length, std::rand);

    int *d_data, *d_perm;
    CUDA_CHK(cudaMalloc(&d_data, sizeof(int)*length));
    CUDA_CHK(cudaMalloc(&d_perm, sizeof(h_perm)));


    Metric m1, m2;
    int errors = false;
    for (auto i = 0; i < BENCH_TIMES && !errors; ++i) {
        std::generate(h_perm.begin(), h_perm.end(), []{
            return rand() % block_size;
        });

        CUDA_CHK(cudaMemcpy(d_perm, h_perm.data(), sizeof(h_perm), cudaMemcpyHostToDevice));

        {
            CUDA_CHK(cudaMemcpy(d_data, h_data, sizeof(int)*length, cudaMemcpyHostToDevice));
            auto t = m1.track_begin();
            block_perm<<<grid_size, block_size>>>(d_data, d_perm, length);
            CUDA_CHK(cudaGetLastError());
            CUDA_CHK(cudaDeviceSynchronize());
            m1.track_end(t);
            CUDA_CHK(cudaMemcpy(h_res1, d_data, sizeof(int)*length, cudaMemcpyDeviceToHost));
        }

        {
            CUDA_CHK(cudaMemcpy(d_data, h_data, sizeof(int)*length, cudaMemcpyHostToDevice));
            auto t = m2.track_begin();
            block_perm_shm<<<grid_size, block_size, shared_mem_size>>>(d_data, d_perm, length);
            CUDA_CHK(cudaGetLastError());
            CUDA_CHK(cudaDeviceSynchronize());
            m2.track_end(t);
            CUDA_CHK(cudaMemcpy(h_res2, d_data, sizeof(int)*length, cudaMemcpyDeviceToHost));
        }

        errors = gpu_compare_arrays(d_data, d_data, length);
    }

    printf("(length=2^%d) Simple Kernel: mean=%f ms, stdev=%f, CV=%f\n", LenLog2, m1.mean(), m1.stdev(), m1.cv());
    printf("(length=2^%d) SHM Kernel   : mean=%f ms, stdev=%f, CV=%f\n", LenLog2, m2.mean(), m2.stdev(), m2.cv());
    printf("Cmp Result: %d\n", errors);

    cudaFree(d_perm);
    cudaFree(d_data);
    delete[] h_res2;
    delete[] h_res1;
    delete[] h_data;
}


void ejercicio3() {
    ejercicio3_impl<24>();
    ejercicio3_impl<26>();
}

#endif // EJERCICIO3_CUH__
