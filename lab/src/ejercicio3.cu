/*!
 * \file ejercicio3.cu
 * \author Juan Ramirez (juan.ramirez@fing.edu.uy)
 */

#include "lab.hpp"
#include "csr_matrix.hpp"

#include <cusparse.h>
#include <vector>

// borrowed from the docs
#define ERR_NE(X,Y) do { \
    if ((X) != (Y)) { \
        fprintf(stderr,"Error in %s at %s:%d\n",__func__,__FILE__,__LINE__); \
        exit(-1);\
    } \
} while(0)

#define CUSPARSE_CALL(X) ERR_NE((X), CUSPARSE_STATUS_SUCCESS)

#ifndef CUSPARSE_SPMV_CSR_ALG1
#define CUSPARSE_SPMV_CSR_ALG1 CUSPARSE_CSRMV_ALG1
#endif

static void do_csr_spmv(size_t rows,
                        size_t columns,
                        const int *d_rowPtr,
                        const int *d_colIdx,
                        const value_type *d_vals,
                        size_t nvals,
                        const value_type *d_vecX,
                        value_type *d_vecY,
                        Metric *m = nullptr)
{
    static constexpr auto alpha = 1.0F;
    static constexpr auto beta = 0.0F;

    cusparseHandle_t handle;
    CUSPARSE_CALL(cusparseCreate(&handle));

    cusparseSpMatDescr_t matDescr;
    CUSPARSE_CALL(cusparseCreateCsr(&matDescr,
                                    rows,
                                    columns,
                                    nvals,
                                    const_cast<int*>(d_rowPtr),
                                    const_cast<int*>(d_colIdx),
                                    const_cast<value_type*>(d_vals),
                                    CUSPARSE_INDEX_32I,
                                    CUSPARSE_INDEX_32I,
                                    CUSPARSE_INDEX_BASE_ZERO,
                                    cuda_data_type<value_type>::type));

    cusparseDnVecDescr_t vecDescrX;
    CUSPARSE_CALL(cusparseCreateDnVec(&vecDescrX,
                                      columns,
                                      const_cast<value_type*>(d_vecX),
                                      cuda_data_type<value_type>::type));

    cusparseDnVecDescr_t vecDescrY;
    CUSPARSE_CALL(cusparseCreateDnVec(&vecDescrY,
                                      rows,
                                      d_vecY,
                                      cuda_data_type<value_type>::type));

    size_t bufferSize;
    CUSPARSE_CALL(cusparseSpMV_bufferSize(handle,
                                          CUSPARSE_OPERATION_NON_TRANSPOSE,
                                          &alpha,
                                          matDescr,
                                          vecDescrX,
                                          &beta,
                                          vecDescrY,
                                          cuda_data_type<value_type>::type,
                                          CUSPARSE_SPMV_CSR_ALG1,
                                          &bufferSize));

    auto d_externalBuffer = dev_alloc<>(bufferSize);


    timeval t;
    if (m) {
        t = m->track_begin();
    }

    CUSPARSE_CALL(cusparseSpMV(handle,
                               CUSPARSE_OPERATION_NON_TRANSPOSE,
                               &alpha,
                               matDescr,
                               vecDescrX,
                               &beta,
                               vecDescrY,
                               cuda_data_type<value_type>::type,
                               CUSPARSE_SPMV_CSR_ALG1,
                               d_externalBuffer.get()));
    CUDA_CHK(cudaDeviceSynchronize());

    if (m) {
        m->track_end(t);
    }

    cusparseDestroyDnVec(vecDescrY);
    cusparseDestroyDnVec(vecDescrX);
    cusparseDestroySpMat(matDescr);
    cusparseDestroy(handle);
}

static void initial_kindergarten_test()
{
    constexpr auto rows = 6LL;
    constexpr auto cols = 5LL;

    std::vector<value_type> h_vals{{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}};
    std::vector<int> h_colIdx{{0, 1, 2, 4, 1, 2, 3, 4, 3, 1, 4}};
    std::vector<int> h_rowPtr{{0, 1, 4, 4, 8, 9, 11}}; // N+1 elements (last is out of vals's bounds)
    value_type h_vecX[cols] = {1, 2, 1, 1, 1};
    value_type h_expectedResult[rows] = {1, 11, 0, 31, 9, 31}; // for X = {1, 2, 1, 1, 1}

    auto d_rowPtr = dev_alloc_fill(h_rowPtr.size(), h_rowPtr.data());
    auto d_colIdx = dev_alloc_fill(h_colIdx.size(), h_colIdx.data());
    auto d_vals = dev_alloc_fill(h_vals.size(), h_vals.data());
    auto d_vecX = dev_alloc_fill(cols, h_vecX);
    auto d_vecY = dev_alloc_zero<value_type>(rows);
    auto d_expectedResult = dev_alloc_fill(rows, h_expectedResult);

    do_csr_spmv(rows,
                cols,
                d_rowPtr.get(),
                d_colIdx.get(),
                d_vals.get(),
                h_vals.size(),
                d_vecX.get(),
                d_vecY.get());

    if (auto ndiff = gpu_compare_arrays(d_expectedResult.get(), d_vecY.get(), rows)) {
        std::cout << "Error: " << ndiff << " differences found" << std::endl;
        value_type h_vecY[rows];
        CUDA_CHK(cudaMemcpy(h_vecY, d_vecY.get(), sizeof(h_vecY), cudaMemcpyDeviceToHost));
        std::cout << "Got: ";
        for (auto i = 0u; i < rows; ++i) {
            std::cout << h_vecY[i] << " ";
        }
        std::cout << std::endl;
    }
}


void ejercicio3()
{
    initial_kindergarten_test();

    lab_tests_controller([&](float p, uint32_t rows, uint32_t columns) {
        CSRMatrix<value_type> mat{rows, columns};
        mat.random_init(p);

        std::vector<value_type> h_vecX(columns, static_cast<value_type>(1));

        auto d_rowPtr = dev_alloc_fill(mat.row_pointers.size(), mat.row_pointers.data());
        auto d_colIdx = dev_alloc_fill(mat.col_indices.size(), mat.col_indices.data());
        auto d_vals = dev_alloc_fill(mat.values.size(), mat.values.data());
        auto d_vecX = dev_alloc_fill(h_vecX.size(), h_vecX.data());
        auto d_vecY = dev_alloc_zero<value_type>(rows);

        Metric m;
        for (auto i = 0; i < BENCH_TIMES; ++i) {
            //auto t = m.track_begin();
            do_csr_spmv(rows,
                        columns,
                        (int*)d_rowPtr.get(),
                        (int*)d_colIdx.get(),
                        d_vals.get(),
                        mat.values.size(),
                        d_vecX.get(),
                        d_vecY.get(),
                        &m);
            //m.track_end(t);
        }
        printf("> %f ms, stdev: %f, CV: %f\n", m.mean(), m.stdev(), m.cv());
    });
}
