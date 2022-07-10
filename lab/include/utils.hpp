/*!
 * \file utils.hpp
 * \author Juan Ramirez (juan.ramirez@fing.edu.uy)
 */
#ifndef UTILS_HPP__
#define UTILS_HPP__

//#include <cuda.h>
#include <assert.h>
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>

#include <unistd.h>
#include <sys/time.h>

#include <cmath>
#include <cstdint>
#include <string>
#include <iostream>
#include <memory>

#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include <cusparse.h>

// should use cudaDeviceProp.warpSize but it is unlikely to change
#define WARP_SIZE (32u)

#define ARRAY_SIZE(arr) (sizeof(arr) / sizeof(arr[0]))

#define CUDA_CHK(ans) gpuAssert((ans), __FILE__, __LINE__)

#define PRINT_DIM(dim)                                                                             \
    do {                                                                                           \
        printf("%s: %d, %d, %d\n", #dim, dim.x, dim.y, dim.z);                                     \
    } while (0)

#define CUDA_MEASURE_START()                                                                       \
    cudaEvent_t measure_start_evt, measure_stop_evt;                                               \
    CUDA_CHK(cudaEventCreate(&measure_start_evt));                                                 \
    CUDA_CHK(cudaEventCreate(&measure_stop_evt));                                                  \
    CUDA_CHK(cudaEventRecord(measure_start_evt, 0))

#define CUDA_MEASURE_STOP(elapTimeVar)                                                             \
    CUDA_CHK(cudaEventRecord(measure_stop_evt, 0));                                                \
    CUDA_CHK(cudaEventSynchronize(measure_stop_evt));                                              \
    CUDA_CHK(cudaEventElapsedTime(&elapTimeVar, measure_start_evt, measure_stop_evt));             \
    CUDA_CHK(cudaEventDestroy(measure_start_evt));                                                 \
    CUDA_CHK(cudaEventDestroy(measure_stop_evt))

template <typename Container>
size_t size_in_bytes(const Container& c) {
    return sizeof(typename Container::value_type) * c.size();
}

struct cuda_deleter {
    void operator()(void* p) {
        cudaFree(p);
    }
};

template<typename T>
using cuda_unique_ptr = std::unique_ptr<T, cuda_deleter>;

template <typename T = uint8_t>
cuda_unique_ptr<T> dev_alloc(size_t size) {
    T *dev_data;
    CUDA_CHK(cudaMalloc(&dev_data, size * sizeof(T)));
    return cuda_unique_ptr<T>(dev_data);
}

template <typename T>
cuda_unique_ptr<T> dev_alloc_zero(size_t size) {
    T *dev_data;
    CUDA_CHK(cudaMalloc(&dev_data, size * sizeof(T)));
    CUDA_CHK(cudaMemset(dev_data, 0, size * sizeof(T)));
    return cuda_unique_ptr<T>(dev_data);
}

template <typename T>
cuda_unique_ptr<T> dev_alloc_fill(size_t size, T *host_data) {
    T *dev_data;
    CUDA_CHK(cudaMalloc(&dev_data, size * sizeof(T)));
    CUDA_CHK(cudaMemcpy(dev_data, host_data, size * sizeof(T), cudaMemcpyHostToDevice));
    return cuda_unique_ptr<T>(dev_data);
}

// https://stackoverflow.com/a/37569519/1351465
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
#else
__device__ inline double atomicAdd(double* address, double val)
{
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}
#endif

/*!
 * Generate pseudo-random numbers between 0 and 1
 */
inline double rand_unif() {
    return static_cast<double>(rand()) / static_cast<double>(RAND_MAX);
}

/*!
 * \brief Like `ceil()` but constexpr
 */
constexpr uint32_t ceilx(double num)
{
    return (static_cast<double>(static_cast<uint32_t>(num)) == num)
        ? static_cast<uint32_t>(num)
        : static_cast<uint32_t>(num) + ((num > 0) ? 1 : 0);
}

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s (%d) at %s:%d\n", cudaGetErrorString(code), code, file,
                line);
        if (abort)
            exit(code);
    }
}

inline void print_device_capabilities() {
    struct _ConvertSMVer2Cores { // borrowed from cuda_helper.h
        int operator()(int major, int minor) {
            typedef struct {
                int SM, Cores;
            } sSMtoCores;
            sSMtoCores nGpuArchCoresPerSM[] = {{0x30, 192}, {0x32, 192}, {0x35, 192}, {0x37, 192},
                                               {0x50, 128}, {0x52, 128}, {0x53, 128}, {0x60, 64},
                                               {0x61, 128}, {0x62, 128}, {0x70, 64},  {0x72, 64},
                                               {0x75, 64},  {-1, -1}};
            for (int index = 0; nGpuArchCoresPerSM[index].SM != -1; ++index) {
                if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor)) {
                    return nGpuArchCoresPerSM[index].Cores;
                }
            }
            return -1;
        }
    } smVer2Cores;

    int deviceID;
    cudaDeviceProp props;
    CUDA_CHK(cudaGetDevice(&deviceID));
    CUDA_CHK(cudaGetDeviceProperties(&props, deviceID));
    printf("Device capability: %d.%d, MPs: %d, Warp Size: %d\n", props.major, props.minor,
           smVer2Cores(props.major, props.minor) * props.multiProcessorCount, props.warpSize);
}

inline uint32_t read_file(const char *fname, int **input) {
    FILE *f = fopen(fname, "r");
    if (f == NULL) {
        fprintf(stderr, "Error: Could not find %s file \n", fname);
        exit(1);
    }

    fseek(f, 0, SEEK_END);
    size_t length = ftell(f);
    fseek(f, 0, SEEK_SET);

    int *buff = (int *)malloc(length * sizeof(int));
    *input = buff;

    int c;
    while ((c = getc(f)) != EOF) {
        *(buff++) = c;
    }

    fclose(f);
    return length;
}

template <typename T> inline std::string val2string(T val, const char *format, int width) {
    char buffer[32] = {0};
    snprintf(buffer, sizeof(buffer), format, val);
    std::string str = buffer;
    std::string padding(width - str.size(), ' ');
    return padding + str;
}

template <typename T>
__host__ __device__ inline bool cmp_equal(T a, T b) {
    return a == b;
}

// to cope with floating-point errors
template <>
__host__ __device__ inline bool cmp_equal<float>(float a, float b) {
    return abs(a - b) < 0.1;
}

// to cope with floating-point errors
template <>
__host__ __device__ inline bool cmp_equal<double>(double a, double b) {
    return abs(a - b) < 0.1;
}

/*!
 * Kernel that compares two arrays and sums the differences
 */
template <typename T> __global__ void cmp_kernelT(T *data1, T *data2, uint32_t length, uint32_t *ndiff) {
    auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < length && !cmp_equal(data1[idx], data2[idx])) {
        //printf("idx=%d, data1=%f, data2=%f\n", idx, data1[idx], data2[idx]);
        atomicAdd(ndiff, 1);
    }
}

/*!
 * Helper function that compares two gpu arrays and returns the number
 * of different elements
 */
template <typename T> uint32_t gpu_compare_arrays(T *a, T *b, uint32_t size) {
    uint32_t h_diff;
    uint32_t *d_diff;
    CUDA_CHK(cudaMalloc(&d_diff, sizeof(uint32_t)));
    CUDA_CHK(cudaMemset(d_diff, 0, sizeof(uint32_t)));
    cmp_kernelT<T><<<std::ceil((double)size / 1024), 1024>>>(a, b, size, d_diff);
    CUDA_CHK(cudaGetLastError());
    CUDA_CHK(cudaMemcpy(&h_diff, d_diff, sizeof(uint32_t), cudaMemcpyDeviceToHost));
    CUDA_CHK(cudaFree(d_diff));
    return h_diff;
}

template <typename T>
void print_vector(const char *name, const T *vect, size_t size) {
    std::cout << name << ": (";
    for (uint32_t i = 0; i < size; ++i) {
        std::cout << vect[i];
        if (i < size-1)
            std::cout << ", ";
    }
    std::cout << ")" << std::endl;
}

template <typename value_type>
bool validate_results(const char *name,
                      value_type *d_expected,
                      value_type *d_out,
                      uint32_t N,
                      bool silent = false,
                      bool print_vect = true)
{
    if (auto ndiff = gpu_compare_arrays(d_expected, d_out, N)) {
        value_type *h_vecY = new value_type[N];
        cudaMemcpy(h_vecY, d_out, sizeof(h_vecY), cudaMemcpyDeviceToHost);
        std::cout << name << ": " << ndiff << " differences found" << std::endl;
        if (print_vect) {
            print_vector("Result", h_vecY, N);
        }
        delete[] h_vecY;
        return false;
    } else {
        if (!silent) {
            std::cout << name << ": OK" << std::endl;
        }
        return true;
    }
}

#endif // UTILS_HPP__
