
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <sys/timeb.h>

#define N 512
#define BENCH2_RUNS 100000000

#ifdef __cplusplus
template <size_t S>
struct Matrix {
  double A[S][S];

  double sum_all() {
    double sum = 0;
    double (&X)[S][S] = A;
    for (size_t i = 0; i < S; ++i) {
      for (size_t j = 0; j < S; ++j) {
        sum += X[i][j];
      }
    }
    return sum;
  }
};
#endif

#define BENCH2_RUN(f) do {                                                            \
        double avg=0;                                                                 \
        double total=0;                                                               \
        volatile double acc=0;                                                                 \
        struct timeval t_ini,t_fin;                                                   \
        gettimeofday(&t_ini, NULL);                                                   \
        for (size_t i = 0; i < BENCH2_RUNS; ++i) {                                    \
            f;                                                                 \
            (void)acc; \
        }                                                                             \
        gettimeofday(&t_fin, NULL);                                                   \
        total = ((double) t_fin.tv_sec * 1000.0 + (double) t_fin.tv_usec / 1000.0 -   \
                ((double) t_ini.tv_sec * 1000.0 + (double) t_ini.tv_usec / 1000.0));  \
        avg = total/(double)BENCH2_RUNS;                                             \
        printf("total: %.03f ms, average: %.09f ms\n", total, avg);                    \
    } while (0)


double suma_est_fil(double A[N][N]) {
  double sum = 0;
  for (size_t i = 0; i < N; ++i) {
    for (size_t j = 0; j < N; ++j) {
      sum += A[i][j];
    }
  }
  return sum;
}

void rand_init(double A[N][N]) {
  for (size_t i = 0; i < N; ++i) {
      for (size_t j = 0; j < N; ++j) {
        A[i][j] = (double)rand();
      }
  }
}


int main() {

#ifdef __cplusplus
  printf("C++\n");
  Matrix<512> A;
  rand_init(A.A);
  BENCH2_RUN(A.sum_all());
  BENCH2_RUN(suma_est_fil(A.A));
#else
  printf("C\n");
  double A[N][N];
  rand_init(A);
  BENCH2_RUN(suma_est_fil(A));
#endif

  return 0;
}
