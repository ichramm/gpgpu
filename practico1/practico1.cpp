#include <cstddef>
#include <cstdlib>
#include <cstdio>
#include "bench.h"

template <size_t Dim>
struct Matrix {
	double A[Dim][Dim];

	void rand_init() {
		for (size_t i = 0; i < Dim; ++i) {
			for (size_t j = 0; j < Dim; ++j) {
				A[i][j] = (double)rand();
			}
		}
	}

	double suma_est_fil() const noexcept {
		double sum = 0;
		for (size_t i = 0; i < Dim; ++i) {
			for (size_t j = 0; j < Dim; ++j) {
				sum += A[i][j];
			}
		}
		return sum;
	}

	double suma_est_col() const noexcept {
		double sum = 0;
		for (size_t j = 0; j < Dim; ++j) {
			for (size_t i = 0; i < Dim; ++i) {
				sum += A[i][j];
			}
		}
		return sum;
	}

};

int main(int argc, char* argv[]) {
	Matrix<512> m;
	m.rand_init();


#if 1
	BENCH2_RUN(m.suma_est_fil(), t_suma_est_fil, t_suma_est_fil_runs);
	BENCH2_RUN(m.suma_est_col(), t_suma_est_col, t_suma_est_col_runs);
	//BENCH_RUN( suma_fil     (A,n)  , t_suma_fil    ,  t_suma_fil_runs     )
	//BENCH_RUN( suma_col     (A,n)  , t_suma_col    ,  t_suma_col_runs     )
	//ENCH_RUN( suma_rand    (A,n)  , t_suma_rand   ,  t_suma_rand_runs    )

	printf("suma_est_fil: %.5f ms\truns:%f\n", t_suma_est_fil, t_suma_est_fil_runs);
	printf("suma_est_col: %.5f ms\truns:%f\n", t_suma_est_col, t_suma_est_col_runs);
	//printf("suma_fila: %.3f ms\truns:%d\n"    , t_suma_fil    ,t_suma_fil_runs     );
	//printf("suma_col: %.3f ms\truns:%d\n"     , t_suma_col    ,t_suma_col_runs     );
	//printf("suma_rand: %.3f ms\truns:%d\n"    , t_suma_rand   ,t_suma_rand_runs    );
#endif
}
