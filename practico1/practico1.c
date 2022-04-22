#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <sys/timeb.h>

#ifndef N
# define N 512
#endif

#ifndef VALT
# define VALT double
#endif

#ifndef BENCH_MS
# define BENCH_MS 2000
#endif

// if no test is selected then we run everything
#if !defined(SUMA_EST_FIL) && \
    !defined(SUMA_EST_COL) && \
    !defined(SUMA_FIL) && \
    !defined(SUMA_COL) && \
    !defined(SUMA_RAND) && \
    !defined(MULT_SIMPLE) && \
    !defined(MULT_FILA) && \
    !defined(MULT_BL_SIMPLE) && \
    !defined(MULT_BL_FILA)
#define SUMA_EST_FIL
#define SUMA_EST_COL
#define SUMA_FIL
#define SUMA_COL
#define SUMA_RAND
#define MULT_SIMPLE
#define MULT_FILA
#define MULT_BL_SIMPLE
#define MULT_BL_FILA
#endif

#if defined(SUMA_FIL) || \
    defined(SUMA_COL) || \
    defined(SUMA_RAND) || \
    defined(MULT_SIMPLE) || \
    defined(MULT_FILA) || \
    defined (MULT_BL_SIMPLE) || \
    defined(MULT_BL_FILA)
#define USE_DYNAMIC_MATRICES
#endif

#define BENCH_RUN(name, fn) do { \
    int acc = 0; /* so function is not optimized away */ \
    size_t iterations = 20; \
    double total_duration_ms = 0; \
    double average_duration_ms = 0;\
    /* will estimate the number of iterations dor a BENCH_MS duration */ \
    printf("%s:\n > estimating function duration with %ld iterations\n", name, iterations); \
    struct timeval t_ini, t_fin; \
    gettimeofday(&t_ini, NULL); \
    for (size_t i = 0; i < iterations; ++i) { \
        acc += fn > 0 ? 1 : 0; \
    } \
    gettimeofday(&t_fin, NULL); \
    total_duration_ms = ((double) t_fin.tv_sec * 1000.0 + (double) t_fin.tv_usec / 1000.0 - \
                        ((double) t_ini.tv_sec * 1000.0 + (double) t_ini.tv_usec / 1000.0)); \
    printf(" > function duration estimated in %.5f milliseconds\n", total_duration_ms/iterations); \
    size_t bench_iterations = (size_t) (BENCH_MS * iterations * 1.1 / total_duration_ms); \
    if (bench_iterations < iterations) { \
        printf(" > function takes too long, skipping benchmark\n"); \
    } else { \
        iterations = bench_iterations; \
        printf(" > benchmarking with %ld iterations (upsampling by 1.1)\n", iterations); \
        /* now begins the actual benchmark */ \
        gettimeofday(&t_ini, NULL); \
        for (size_t i = 0; i < iterations; ++i) { \
            acc += fn > 0 ? 1 : 0; \
        } \
        gettimeofday(&t_fin, NULL); \
        total_duration_ms = ((double) t_fin.tv_sec * 1000.0 + (double) t_fin.tv_usec / 1000.0 - \
                            ((double) t_ini.tv_sec * 1000.0 + (double) t_ini.tv_usec / 1000.0)); \
    } \
    average_duration_ms = total_duration_ms / iterations; \
    printf(" > total_duration_ms: %f ms, average_duration_ms: %.5f ms\n",  \
           total_duration_ms, average_duration_ms, acc); \
} while(0)

VALT suma_est_fil(VALT  A[N][N]);
VALT suma_est_col(VALT  A[N][N]);

VALT suma_fil(VALT* A, size_t n);
VALT suma_col(VALT* A, size_t n);
VALT suma_rand(VALT* A, size_t n);

int mult_simple(const VALT* __restrict__ A, const VALT* __restrict__ B, VALT* __restrict__ C, size_t n);
int mult_fila(const VALT* __restrict__ A, const VALT* __restrict__ B, VALT* __restrict__ C, size_t n);
int mult_bl_simple(const VALT* __restrict__ A, const VALT* __restrict__ B, VALT* __restrict__ C, size_t n, size_t bl_sz);
int mult_bl_fila(const VALT* __restrict__ A, const VALT* __restrict__ B, VALT* __restrict__ C, size_t n, size_t bl_sz);

void random_vector(VALT* a, size_t n) {
    for (unsigned int i = 0; i < n; i++) {
        a[i] = (float)rand() / (float)RAND_MAX;
    }
}

int main(int argc __attribute__((unused)),
         char* argv[] __attribute__((unused))) {
    srand(35141);

#ifdef USE_DYNAMIC_MATRICES
    if (argc < 2) {
        printf("El programa recibe n, y nb: n es la dimensión de las matrices y nb es el tamaño de bloque\n");
        exit(1);
    }
    int n = atoi(argv[1]);
    int nb = atoi(argv[2]);
    (void)nb; // hack to disable "unused variable" warning

    VALT* A = (VALT*)aligned_alloc(64, n * n * sizeof(VALT));
    VALT* B = (VALT*)aligned_alloc(64, n * n * sizeof(VALT));
    VALT* C = (VALT*)aligned_alloc(64, n * n * sizeof(VALT));
    random_vector(A, n * n);
    random_vector(B, n * n);
#endif

    VALT A_est[N][N];

    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            A_est[i][j] = (VALT)rand();

#ifdef SUMA_EST_FIL
        BENCH_RUN("suma_est_fil", suma_est_fil(A_est));
#endif
#ifdef SUMA_EST_COL
        BENCH_RUN("suma_est_col", suma_est_col(A_est));
#endif
#ifdef SUMA_FIL
        BENCH_RUN("suma_fil", suma_fil(A, n));
#endif
#ifdef SUMA_COL
        BENCH_RUN("suma_col", suma_col(A, n));
#endif
#ifdef SUMA_RAND
        BENCH_RUN("suma_rand", suma_rand(A, n));
#endif
#ifdef MULT_SIMPLE
        BENCH_RUN("mult_simple", mult_simple(A, B, C, n));
#endif
#ifdef MULT_FILA
        BENCH_RUN("mult_fila", mult_fila(A, B, C, n));
#endif
#ifdef MULT_BL_SIMPLE
        BENCH_RUN("mult_bl_simple", mult_bl_simple(A, B, C, n, nb));
#endif
#ifdef MULT_BL_FILA
        BENCH_RUN("mult_bl_fila", mult_bl_fila(A, B, C, n, nb));
#endif

#ifdef USE_DYNAMIC_MATRICES
    free(A);
    free(B);
    free(C);
#endif

    return 0;
}


VALT suma_est_fil(VALT A[N][N]) {
    VALT sum = 0;
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            sum += A[i][j];
        }
    }
    return sum;
}

VALT suma_est_col(VALT A[N][N]) {
    VALT sum = 0;
    for (size_t j = 0; j < N; ++j) {
        for (size_t i = 0; i < N; ++i) {
            sum += A[i][j];
        }
    }
    return sum;
}


VALT suma_fil(VALT* A, size_t n) {
    VALT sum = 0;
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            sum += A[i * n + j];
        }
    }
    return sum;
}

VALT suma_col(VALT* A, size_t n) {
    VALT sum = 0;
    for (size_t j = 0; j < n; ++j) {
        for (size_t i = 0; i < n; ++i) {
            sum += A[i * n + j];
        }
    }
    return sum;
}

VALT suma_rand(VALT* A, size_t n) {
    VALT sum = 0;
    for (size_t i = 0; i < n * n; ++i) {
        // won't return the correct value, but it's ok for the benchmark
        size_t idx = rand() % (n * n);
        sum += A[idx];
    }
    return sum;
}

int mult_simple(const VALT* __restrict__ A, const VALT* __restrict__ B, VALT* __restrict__ C, size_t n) {
    for (size_t row = 0; row < n; ++row) {
        for (size_t col = 0; col < n; ++col) {
            C[row * n + col] = 0;
            for (size_t k = 0; k < n; ++k) {
                C[row * n + col] += A[row * n + k] * B[k * n + col];
            }
        }
    }
    return 0;
}

// pre: C must be zeroed
int mult_fila(const VALT* __restrict__ A, const VALT* __restrict__ B, VALT* __restrict__ C, size_t n) {
#if 0
    -- - primera fila de a, primera fila de b
        c(1, 1) += a(1, 1) * b(1, 1)
        c(1, 2) += a(1, 1) * b(1, 2)
        c(1, 3) += a(1, 1) * b(1, 3)
        -- - primera fila de a, segunda fila de b
        c(1, 1) += a(1, 2) * b(2, 1)
        c(1, 2) += a(1, 2) * b(2, 2)
        c(1, 3) += a(1, 2) * b(2, 3)
        -- - primera fila de a, tercera fila de b
        c(1, 1) += a(1, 3) * b(3, 1)
        c(1, 2) += a(1, 3) * b(3, 2)
        c(1, 3) += a(1, 3) * b(3, 3)
#endif
        for (size_t a_row = 0; a_row < n; ++a_row) {
            for (size_t b_row = 0; b_row < n; ++b_row) {
                for (size_t col = 0; col < n; ++col) {
                    C[a_row * n + col] += A[a_row * n + b_row] * B[b_row * n + col];
                }
            }
        }

    return 0;
}

// pre: C must be zeroed
int mult_bl_simple(const VALT* __restrict__ A, const VALT* __restrict__ B, VALT* __restrict__ C, size_t n, size_t bl_sz) {
    for (size_t blk_i = 0; blk_i < n / bl_sz; ++blk_i) {
        for (size_t blk_j = 0; blk_j < n / bl_sz; ++blk_j) {
            // will set block C[blk_i, blk_j]
            size_t c_start = blk_i * bl_sz * n + blk_j * bl_sz;
            for (size_t blk_k = 0; blk_k < n / bl_sz; ++blk_k) {
                // will compute A[blk_i, blk_k] * B[blk_k, blk_j]
                size_t a_start = blk_i * bl_sz * n + blk_k * bl_sz;
                size_t b_start = blk_k * bl_sz * n + blk_j * bl_sz;

                for (size_t i = 0; i < bl_sz; ++i) {
                    for (size_t j = 0; j < bl_sz; ++j) {
                        for (size_t k = 0; k < bl_sz; ++k) {
                            C[c_start + (i * n) + j] += A[a_start + (i * n) + k] * B[b_start + j + (k * n)];
                        }
                    }
                }
            }
        }
    }

    return 0;
}

// pre: C must be zeroed
int mult_bl_fila(const VALT* __restrict__ A, const VALT* __restrict__ B, VALT* __restrict__ C, size_t n, size_t bl_sz) {
    for (size_t blk_i = 0; blk_i < n/bl_sz; ++blk_i) {
        for (size_t blk_j = 0; blk_j < n/bl_sz; ++blk_j) {
            // C[blk_i, blk_j]
            size_t c_start = blk_i*bl_sz*n + blk_j*bl_sz;
            for (size_t blk_k = 0; blk_k < n/bl_sz; ++blk_k) {
                size_t a_start = blk_i*bl_sz*n + blk_k*bl_sz; // A[blk_i, blk_k]
                size_t b_start = blk_k*bl_sz*n + blk_j*bl_sz; // B[blk_k, blk_j]

                for (size_t a_row = 0; a_row < bl_sz; ++a_row) {
                    for (size_t b_row = 0; b_row < bl_sz; ++b_row) {
                        for (size_t col = 0; col < bl_sz; ++col) {
                            C[c_start + a_row * n + col] += A[a_start + a_row * n + b_row] * B[b_start + b_row * n + col];
                        }
                    }
                }
            }
        }
    }

    return 0;
}
