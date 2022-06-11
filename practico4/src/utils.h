/*!
 * \file utils.h
 * \author Juan Ramirez (juan.ramirez@fing.edu.uy)
 */
#ifndef UTILS_H__
#define UTILS_H__

//#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <ctype.h>

#include <unistd.h>
#include <sys/time.h>

#include <cstdint>
#include <string>

#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>

// should use cudaDeviceProp.warpSize but it is unlikely to change
#define WARP_SIZE (32u)

#define ARRAY_SIZE(arr) (sizeof(arr)/sizeof(arr[0]))

#define CUDA_CHK(ans) gpuAssert((ans), __FILE__, __LINE__)

#define PRINT_DIM(dim) do {\
    printf("%s: %d, %d, %d\n", #dim, dim.x, dim.y, dim.z); \
} while(0)

#define CUDA_MEASURE_START() \
    cudaEvent_t measure_start_evt, measure_stop_evt; \
    CUDA_CHK(cudaEventCreate(&measure_start_evt)); \
    CUDA_CHK(cudaEventCreate(&measure_stop_evt));  \
    CUDA_CHK(cudaEventRecord(measure_start_evt, 0))

#define CUDA_MEASURE_STOP(elapTimeVar) \
    CUDA_CHK(cudaEventRecord(measure_stop_evt, 0)); \
    CUDA_CHK(cudaEventSynchronize(measure_stop_evt)); \
    CUDA_CHK(cudaEventElapsedTime(&elapTimeVar, measure_start_evt, measure_stop_evt)); \
    CUDA_CHK(cudaEventDestroy(measure_start_evt)); \
    CUDA_CHK(cudaEventDestroy(measure_stop_evt))

struct Metric
{
    size_t count_ = 0;
    double total_ = 0;
    double sumsq_ = 0;

    timeval track_begin() {
        timeval tv;
        gettimeofday(&tv, NULL);
        return tv;
    }

    void track_end(timeval tv_start) {
        timeval tv_end;
        gettimeofday(&tv_end,NULL);
        double elap =  ((double) tv_end.tv_sec * 1000.0 + (double) tv_end.tv_usec / 1000.0 -
                       ((double) tv_start.tv_sec * 1000.0 + (double) tv_start.tv_usec / 1000.0));
        add_sample(elap);
    }

    void add_sample(double sample) {
        count_ += 1;
        total_ += sample;
        sumsq_ += sample * sample;
    }

    double total() const {
        return total_;
    }

    double mean() const {
        return count_ ? total_ / count_ : 0;
    }

    // basado en unidad 1 - sesion 2 del curso Metodos de Monte Carlo
    // https://eva.fing.edu.uy/course/view.php?id=24
    double stdev() const {
        if (count_ > 1) {
            return std::sqrt(sumsq_ / (count_ * (count_ - 1)) - (mean() * mean()) / (count_ - 1));
        }
        return 0;
    }

    double cv() const {
        auto m = mean();
        return m != 0 ? stdev() / m : 0;
    }
};

/*!
 * \brief Returns the minimum multiple of the warp size that is greater
 * than or equal to the given value `n`.
 */
constexpr uint32_t roundup_to_warp_size(uint32_t n)
{
    // works as long as WARP_SIZE is a power of 2
    return (n + WARP_SIZE - 1) & ~(WARP_SIZE - 1);
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

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s (%d) at %s:%d\n", cudaGetErrorString(code), code, file, line);
        if (abort)
            exit(code);
    }
}

inline void print_device_capabilities()
{
    struct _ConvertSMVer2Cores { // borrowed from cuda_helper.h
        int operator()(int major, int minor) {
            typedef struct { int SM, Cores; } sSMtoCores;
            sSMtoCores nGpuArchCoresPerSM[] = {
                {0x30, 192}, {0x32, 192}, {0x35, 192}, {0x37, 192}, {0x50, 128},
                {0x52, 128}, {0x53, 128}, {0x60, 64}, {0x61, 128}, {0x62, 128},
                {0x70, 64}, {0x72, 64}, {0x75, 64}, {-1, -1}};
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
    printf("Device capability: %d.%d, MPs: %d, Warp Size: %d\n",
           props.major,
           props.minor,
           smVer2Cores(props.major, props.minor) * props.multiProcessorCount,
           props.warpSize);
}

static uint32_t read_file(const char * fname, int** input)
{
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

template <typename T>
__global__ void cmp_kernelT(T * data1, T *data2, unsigned int length, int *ndiff) {
    auto idx= blockIdx.x *blockDim.x + threadIdx.x;
    if (idx < length && data1[idx] != data2[idx]) {
        atomicAdd(ndiff, 1);
    }
}

template <typename T>
int gpu_compare_arrays(T *a, T *b, unsigned int size)
{
    int h_diff;
    int *d_diff;
    cudaMalloc(&d_diff, sizeof(int));
    cudaMemset(d_diff, 0, sizeof(int));
    cmp_kernelT<T><<<ceilx(size/1024), 1024>>>(a, b, size, d_diff);
    cudaMemcpy(&h_diff, d_diff, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_diff);
    return h_diff;
}

#endif // UTILS_H__
