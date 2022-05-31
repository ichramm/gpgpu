/*!
 * \file utils.h
 * \author Juan Ramirez (juan.ramirez@fing.edu.uy)
 */
#ifndef UTILS_H__
#define UTILS_H__

//#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <assert.h>
#include <unistd.h>

#include <sys/time.h>

#include <string>

// should use cudaDeviceProp.warpSize but it is unlikely to change
#define WARP_SIZE (32u)

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
    size_t count = 0;
    double sum = 0;
    double sumsq = 0;

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
        count += 1;
        sum += sample;
        sumsq += sample * sample;
    }

    double mean() const {
        return sum / count;
    }

    // basado en unidad 1 - sesion 2 del curso Metodos de Monte Carlo
    // https://eva.fing.edu.uy/course/view.php?id=24
    double stdev() const {
        if (count > 1) {
            return std::sqrt(sumsq / (count * (count - 1)) - (mean() * mean()) / (count - 1));
        }
        return 0;
    }
};

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
constexpr unsigned int ceilx(double num)
{
    return (static_cast<double>(static_cast<unsigned int>(num)) == num)
        ? static_cast<unsigned int>(num)
        : static_cast<unsigned int>(num) + ((num > 0) ? 1 : 0);
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

static unsigned int read_file(const char * fname, int** input)
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

#endif // UTILS_H__
