/*!
 * \file lab.hpp
 * \author Juan Ramirez (juan.ramirez@fing.edu.uy)
 */
#ifndef LAB_HPP__
#define LAB_HPP__

#include "utils.hpp"
#include "metric.hpp"

typedef float value_type;

#ifndef BLOCK_SIZE
#define BLOCK_SIZE (256u)
#endif

#ifndef BENCH_TIMES
#define BENCH_TIMES 100
#endif

#ifndef MAT_ROWS
#define MAT_ROWS (10000u)
#endif

#ifndef MAT_COLS
#define MAT_COLS (10000u)
#endif

template <typename T>
struct cuda_data_type { };

template<> struct cuda_data_type<int> {
    static constexpr cudaDataType type = CUDA_R_32I; // SpMV not supported for int
};
template<> struct cuda_data_type<float> {
    static constexpr cudaDataType type = CUDA_R_32F;
};

template<> struct cuda_data_type<double> {
    static constexpr cudaDataType type = CUDA_R_64F;
};

void ejercicio1();

void ejercicio2();

void ejercicio3();

#endif // LAB_HPP__
