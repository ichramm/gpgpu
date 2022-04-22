#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include "cuda.h"
#include <locale.h>

#define CUDA_CHK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

void read_file(const char*, int*);
size_t get_text_length(const char * fname);

#define A 15
#define B 27
#define M 256
#define A_MMI_M -17

#define N 512

// para la parte que require cantidad fija de bloques
#define BLOCKS 128

#define CUDA_MEASURE_START() \
    cudaEvent_t start, stop; \
    cudaEventCreate(&start); \
    cudaEventCreate(&stop); \
    cudaEventRecord(start, 0);

#define CUDA_MEASURE_STOP(elapTimeVar) \
    cudaEventRecord(stop, 0); \
    cudaEventSynchronize(stop); \
    cudaEventElapsedTime(&elapTimeVar, start, stop); \
    cudaEventDestroy(start); \
    cudaEventDestroy(stop);

__device__ int modulo(int a, int b) {
	int r = a % b;
	r = (r < 0) ? r + b : r;
	return r;
}

/**
 * Kernel de la parte 1.a, funciona solo para los primeros N caracteres
 */
__global__ void decrypt_kernel_1a(int *d_message, size_t length) {
    d_message[threadIdx.x] = modulo(A_MMI_M * (d_message[threadIdx.x] - B), M);
}

/**
 * Kernel de la parte 1.b
 */
__global__ void decrypt_kernel_1b(int *d_message, size_t length) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < length) {
        d_message[idx] = modulo(A_MMI_M * (d_message[idx] - B), M);
    }
}

/**
 * Kernel de la parte 1.c, cada thread debe procesar varios caracteres
 */
__global__ void decrypt_kernel_1c(int *d_message, size_t length) {
    size_t block_txt_size = ceilf(length / (float)gridDim.x);
    size_t block_begin = blockIdx.x * block_txt_size;
    //size_t block_end = block_begin + block_txt_size;

    size_t thread_txt_size = ceilf(block_txt_size / (float)blockDim.x);
    size_t thread_beg = block_begin + threadIdx.x * thread_txt_size;
    size_t thread_end = thread_beg + thread_txt_size;

    for (size_t i = thread_beg; i < thread_end && i < length; i++) {
        d_message[i] = modulo(A_MMI_M * (d_message[i] - B), M);
    }
}

/**
 * Kernel del ejercicio 2
 */
__global__ void count_occurrences(int *d_message, int length, unsigned int *d_counts) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < length) {
        atomicAdd(&(d_counts[d_message[idx]]), 1);
    }
}

/**
 * Idem anterior pero usando memoria compartida
 * Se observa una mejora de 3x
 */
__global__ void count_occurrences_shared_mem(int *d_message, int length, unsigned int *d_counts) {
    __shared__ unsigned int partial_counts[M];
    int msg_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (msg_idx < length) {
        atomicAdd(&(partial_counts[d_message[msg_idx]]), 1);

        __syncthreads();

        if (threadIdx.x == 0) {
            for (int i = 0; i < M; i++) {
                atomicAdd(&(d_counts[i]), partial_counts[i]);
            }
        }
    }
}

/**
 * Solucion del ejercicio 1.a
 */
static int ejercicio1a(const int * const h_message, size_t length) {

    size_t size = length * sizeof(int);
    float elapsedTime;

    int *d_message;
    int * decryped_msg = (int *)malloc(size);
    CUDA_CHK(cudaMalloc(&d_message, size));

    /* copiar los datos de entrada a la GPU */
    CUDA_CHK(cudaMemcpy(d_message, h_message, size, cudaMemcpyHostToDevice));

    /* Configurar la grilla y lanzar el kernel */
    dim3 dimGrid(1, 1, 1);
    dim3 dimBlock(N, 1, 1);
    CUDA_MEASURE_START();
    decrypt_kernel_1a<<<dimGrid, dimBlock>>>(d_message, length);
    CUDA_MEASURE_STOP(elapsedTime);

    /* Copiar los datos de salida a la CPU en h_message (cudaMemcpy ya sincroniza) */
    CUDA_CHK(cudaMemcpy(decryped_msg, d_message, size, cudaMemcpyDeviceToHost));

    // despliego el mensaje (solo los primeros N caracteres)
    for (size_t i = 0; i < N; ++i) {
        printf("%c", (char)decryped_msg[i]);
    }
    printf("\n");

    printf("Elapsed Time: %f ms\n", elapsedTime);

    return 0;
}

/**
 * Solucion del ejercicio 1.b
 */
static int ejercicio1b(const int * const h_message, size_t length) {

    size_t size = length * sizeof(int);
    float elapsedTime;

    int *d_message;
    int * decryped_msg = (int *)malloc(size);
    CUDA_CHK(cudaMalloc(&d_message, size));

    /* copiar los datos de entrada a la GPU */
    CUDA_CHK(cudaMemcpy(d_message, h_message, size, cudaMemcpyHostToDevice));

    /* Configurar la grilla y lanzar el kernel */

    // cantidad de bloques depende del largo del texto
    // cumple con "para que utilice varios bloques, procesando textos de largo arbitrario"
    dim3 dimGrid(length/N+1, 1, 1);
    dim3 dimBlock(N, 1, 1);
    CUDA_MEASURE_START();
    decrypt_kernel_1b<<<dimGrid, dimBlock>>>(d_message, length);
    CUDA_MEASURE_STOP(elapsedTime);

    /* Copiar los datos de salida a la CPU en h_message (cudaMemcpy ya sincroniza) */
    CUDA_CHK(cudaMemcpy(decryped_msg, d_message, size, cudaMemcpyDeviceToHost));

    // despliego el mensaje (solo los primeros N caracteres)
    for (size_t i = 0; i < length; ++i) {
        printf("%c", (char)decryped_msg[i]);
    }
    printf("\n");

    cudaFree(d_message);
    free(decryped_msg);

    printf("Elapsed Time: %f ms\n", elapsedTime);

    return 0;
}

/**
 * Solucion del ejercicio 1.c
 */
static int ejercicio1c(const int * const h_message, size_t length) {

    size_t size = length * sizeof(int);
    float elapsedTime;

    int *d_message;
    int * decryped_msg = (int *)malloc(size);
    CUDA_CHK(cudaMalloc(&d_message, size));

    /* copiar los datos de entrada a la GPU */
    CUDA_CHK(cudaMemcpy(d_message, h_message, size, cudaMemcpyHostToDevice));

    /* Configurar la grilla y lanzar el kernel */
    dim3 dimGrid(BLOCKS, 1, 1);
    dim3 dimBlock(N, 1, 1);
    CUDA_MEASURE_START();
    decrypt_kernel_1c<<<dimGrid, dimBlock>>>(d_message, length);
    CUDA_MEASURE_STOP(elapsedTime);

    /* Copiar los datos de salida a la CPU en h_message (cudaMemcpy ya sincroniza) */
    CUDA_CHK(cudaMemcpy(decryped_msg, d_message, size, cudaMemcpyDeviceToHost));

    // despliego el mensaje (solo los primeros N caracteres)
    for (size_t i = 0; i < length; ++i) {
        printf("%c", (char)decryped_msg[i]);
    }
    printf("\n");

    cudaFree(d_message);
    free(decryped_msg);

    printf("Elapsed Time: %f ms\n", elapsedTime);

    return 0;
}

/**
 * Solucion del ejercicio 2
 */
static int ejercicio2(const int * const h_message, size_t length)
{
    // 1: desencriptar acorde a la parte 1.b
    int *d_message;
    size_t size = length * sizeof(int);
    CUDA_CHK(cudaMalloc(&d_message, size));
    CUDA_CHK(cudaMemcpy(d_message, h_message, size, cudaMemcpyHostToDevice));
    dim3 dimGrid(length/N + 1, 1, 1); // + 1 por si length no es multiplo de N
    dim3 dimBlock(N, 1, 1);
    decrypt_kernel_1b<<<dimGrid, dimBlock>>>(d_message, length);
    CUDA_CHK(cudaDeviceSynchronize());

    // 2: contar las ocurrencias de cada caracter
    unsigned int *d_counts;
    unsigned int *h_counts = (unsigned int *)malloc(M * sizeof(unsigned int));
    CUDA_CHK(cudaMalloc(&d_counts, M * sizeof(unsigned int)));

    float elapsedTimeShared, elapsedTimeSimple;

    {
        dimGrid = dim3(length/1024 + 1, 1, 1); // + 1 por si length no es multiplo de N
        dimBlock = dim3(1024, 1, 1);
        CUDA_CHK(cudaMemset(d_counts, 0, M * sizeof(unsigned int)));
        CUDA_MEASURE_START();
        count_occurrences_shared_mem<<<dimGrid, dimBlock, M * sizeof(unsigned int)>>>(d_message, length, d_counts);
        CUDA_CHK(cudaDeviceSynchronize());
        CUDA_MEASURE_STOP(elapsedTimeShared);
    }

    {
        dimGrid = dim3(length/N + 1, 1, 1); // + 1 por si length no es multiplo de N
        dimBlock = dim3(N, 1, 1);
        CUDA_CHK(cudaMemset(d_counts, 0, M * sizeof(unsigned int)));
        CUDA_MEASURE_START();
        count_occurrences<<<dimGrid, dimBlock>>>(d_message, length, d_counts);
        CUDA_CHK(cudaDeviceSynchronize());
        CUDA_MEASURE_STOP(elapsedTimeSimple);
    }

    CUDA_CHK(cudaMemcpy(h_counts, d_counts, M * sizeof(unsigned int), cudaMemcpyDeviceToHost));

    // 3: imprimir los caracteres que aparecen mas de una vez
    for (int i = 0; i < M; ++i) {
        if (h_counts[i] > 0) {
            // agrego el hexa por los caracteres que no se imprimen correctamente en la consola
            printf("[%c]: %d\n", (char)i, h_counts[i]);
        }
    }

    printf("Elapsed time using shared memory  : %f ms\n", elapsedTimeShared);
    printf("Elapsed time without shared memory: %f ms\n", elapsedTimeSimple);

    free(h_counts);
    cudaFree(d_counts);
    cudaFree(d_message);

    return 0;
}

int main(int argc, char *argv[])
{
	int *h_message;
	unsigned int size;

	const char * fname, *part = NULL;

	if (argc < 2 || strcmp(argv[1], "-h") == 0) {
        printf("Usage: ./practico2 secret.txt [part]\n\t ; part = 1a|1b|1c|2\n");
        return 1;
    }

	fname = argv[1];

    if (argc > 2) {
        part = argv[2];
    }

	size_t length = get_text_length(fname);

	size = length * sizeof(int);

	// reservar memoria para el mensaje
	h_message = (int *)malloc(size);

    // para copiar de la GPU a la CPU sin sobrescribir h_message
    int * h_message2 = (int *)malloc(size);

	// leo el archivo de la entrada
	read_file(fname, h_message);

#define SOLVE_EXERCISE(name, desc, code) do {\
    if (!part || strcmp(part, name) == 0) { \
        printf("---------- Begin %s:\n", desc); \
        code; \
        printf("---------- End %s ----------\n", desc); \
    } \
} while(false)

    SOLVE_EXERCISE("1a", "Ejercicio 1 - Parte a", ejercicio1a(h_message, length));
    SOLVE_EXERCISE("1b", "Ejercicio 1 - Parte b", ejercicio1b(h_message, length));
    SOLVE_EXERCISE("1c", "Ejercicio 1 - Parte c", ejercicio1c(h_message, length));
    SOLVE_EXERCISE("2", "Ejercicio 2", ejercicio2(h_message, length));

	// libero la memoria en la CPU
	free(h_message);

	return 0;
}


size_t get_text_length(const char * fname)
{
	FILE *f = NULL;
	f = fopen(fname, "r"); //read and binary flags

	size_t pos = ftell(f);
	fseek(f, 0, SEEK_END);
	size_t length = ftell(f);
	fseek(f, pos, SEEK_SET);

	fclose(f);

	return length;
}

void read_file(const char * fname, int* input)
{
	// printf("leyendo archivo %s\n", fname );

	FILE *f = NULL;
	f = fopen(fname, "r"); //read and binary flags
	if (f == NULL){
		fprintf(stderr, "Error: Could not find %s file \n", fname);
		exit(1);
	}

	//fread(input, 1, N, f);
	int c;
	while ((c = getc(f)) != EOF) {
		*(input++) = c;
	}

	fclose(f);
}
