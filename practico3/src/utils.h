#ifndef UTILS_H__
#define UTILS_H__

//#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>

#include <string>

// should use cudaDeviceProp.warpSize but it is unlikely to change
#define WARP_SIZE (32u)

#define CUDA_CHK(ans) gpuAssert((ans), __FILE__, __LINE__)

#define CUDA_MEASURE_START()           \
    cudaEvent_t measure_start_evt, measure_stop_evt;           \
    CUDA_CHK(cudaEventCreate(&measure_start_evt)); \
    CUDA_CHK(cudaEventCreate(&measure_stop_evt));  \
    CUDA_CHK(cudaEventRecord(measure_start_evt, 0))

#define CUDA_MEASURE_STOP(elapTimeVar)                         \
    CUDA_CHK(cudaEventRecord(measure_stop_evt, 0));                        \
    CUDA_CHK(cudaEventSynchronize(measure_stop_evt));                      \
    CUDA_CHK(cudaEventElapsedTime(&elapTimeVar, measure_start_evt, measure_stop_evt)); \
    CUDA_CHK(cudaEventDestroy(measure_start_evt));                         \
    CUDA_CHK(cudaEventDestroy(measure_stop_evt))

/*!
 * \brief Returns the minimum multiple of the warp size that is greater
 * than or equal to the given value `n`.
 */
constexpr unsigned int roundup_to_warp_size(unsigned int n)
{
    // works as long as WARP_SIZE is a power of 2
    return (n + WARP_SIZE - 1) & ~(WARP_SIZE - 1);
}

/*!
 * \brief Like `ceil()` but constexpr
 */
constexpr unsigned int ceilx(double x)
{
    unsigned int floor = (unsigned int) x;
    return (x - floor) > 0.0 ? floor + 1 : floor;
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

#endif // UTILS_H__
