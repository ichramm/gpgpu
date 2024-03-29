#include <math.h>
#include <sys/time.h>
#include <sys/timeb.h>
#include <stdlib.h>

#define VALT double
#define STDEV_SAMPLES 50
#define BENCH_MS 2000
#define BENCH2_NRUNS 100000000

#define N 512

#ifdef __cplusplus
extern "C" {
#endif
    void dont_optimize();
#ifdef __cplusplus
}
#endif

#define BENCH_RUN(f,elap,runs)                                                                              \
        double elap=0;                                                                                      \
        int runs=0;                                                                                         \
        {                                                                                                   \
        double total=0;                                                                                     \
        struct timeval t_ini,t_fin;                                                                         \
        while (total < BENCH_MS){                                                                           \
            gettimeofday(&t_ini,NULL);                                                                      \
            f;                                                                                              \
            gettimeofday(&t_fin,NULL);                                                                      \
            double elap =  ((double) t_fin.tv_sec * 1000.0 + (double) t_fin.tv_usec / 1000.0 -              \
                           ((double) t_ini.tv_sec * 1000.0 + (double) t_ini.tv_usec / 1000.0));             \
            total+=elap;                                                                                    \
            runs++;                                                                                         \
        }                                                                                                   \
        elap = total/(double)runs;                                                                          \
        }

#define BENCH2_RUN(f,elap,total)                                                                \
    double elap=0;                                                                              \
    double total=0;                                                                             \
    {                    \
    VALT acc;                                                                        \
        struct timeval t_ini,t_fin;                                                             \
        gettimeofday(&t_ini, NULL);                                                              \
        for (size_t i = 0; i < BENCH2_NRUNS; ++i) {                                              \
            acc += f;                                                                                  \
        }                                                                                       \
        gettimeofday(&t_fin, NULL);                                                              \
        total = ((double) t_fin.tv_sec * 1000.0 + (double) t_fin.tv_usec / 1000.0 -            \
                ((double) t_ini.tv_sec * 1000.0 + (double) t_ini.tv_usec / 1000.0));     \
        elap = total/(double)BENCH2_NRUNS;                                                       \
    }
