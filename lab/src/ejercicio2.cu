/*!
 * \file ejercicio2.cu
 * \author Juan Ramirez (juan.ramirez@fing.edu.uy)
 */
#include <iostream>
#include <algorithm>
#include <functional>

#include "lab.hpp"
#include "bl_matrix.hpp"

template <typename value_type, size_t Block_Width>
__host__ __device__ void build_dense_block(uint32_t j,
                                           const uint64_t * __restrict__ bl_bitmaps,
                                           const uint32_t * __restrict__ bl_starts,
                                           const value_type * __restrict__ values,
                                           value_type *block)
{
    auto bitmap = bl_bitmaps[j];
    auto start = bl_starts[j];
    auto offset = 0u;
    for (auto i = 0u; i < Block_Width*Block_Width; ++i) {
        auto bit = 1ULL << ((Block_Width*Block_Width) - i - 1);
        if (bitmap & bit) {
            block[i] = values[start + offset];
            ++offset;
        }
    }
}

template <typename value_type, size_t Block_Width>
__host__ void serial_bl_spmv_kernel_host(const BLMatrix<value_type, Block_Width>& mat,
                                      const value_type * __restrict__ x,
                                      value_type *y)
{
    for (auto i = 0u; i < mat.rows / Block_Width; ++i) {
        auto block_row = i * Block_Width;
        // j runs though the block columns in the current row
        for (auto j = mat.bl_row_pointers[i]; j < mat.bl_row_pointers[i + 1]; ++j) {
            value_type block[Block_Width * Block_Width] = {0};
            build_dense_block<value_type, Block_Width>(j,
                                                       mat.bl_bitmaps.data(),
                                                       mat.bl_starts.data(),
                                                       mat.values.data(),
                                                       block);

            auto block_col = mat.bl_col_indices[j] * Block_Width;
            for (auto k = 0u; k < Block_Width; ++k) {
                for (auto l = 0u; l < Block_Width; ++l) {
                    y[block_row + k] += block[k * Block_Width + l] * x[block_col + l];
                }
            }
        }
    }
}


// one block per block??
// 2D grid of size (n/Block_Width) x (m/Block_Width)
// problem: number of blocks does not match grid dimensions
// need to consider null blocks
// idea: use nested kernels -> cannot uses shared memory
// but: I kwnow how may blocks there are
// idea: use 8x8 blocks
    // idea: un bloque por fila de bloques
    // usar distintos grupos de threads para cada bloque
    // ver si no hay quilombo al sincronizar
    // buscar la forma de hacer parallel reduce


template <typename value_type, size_t Block_Width>
__global__ void par_bl_spmv_kernel1(typename BLMatrix<value_type, Block_Width>::DeviceStruct mat,
                                 const value_type * __restrict__ vecX,
                                 value_type *vecY)
{
    // one value per row
    __shared__ value_type partial_sum[Block_Width];

    if (threadIdx.x == 0) {
        partial_sum[threadIdx.y & (Block_Width-1)] = 0;
    }

    __syncthreads();

    // 2d grid
    auto block_row = blockIdx.y;
    auto block_offset = blockIdx.x;

    // coordinates inside the block
    auto y = threadIdx.y;
    auto x = threadIdx.x;

    for (auto j = mat.bl_row_pointers[block_row] + block_offset,
              end = mat.bl_row_pointers[block_row + 1]; j < end; j += gridDim.x) {
        auto block_col = mat.bl_col_indices[j] * Block_Width;

        auto bitmap = mat.bl_bitmaps[j];
        auto start = mat.bl_starts[j];
        auto bit = 1ULL << ((Block_Width*Block_Width) - (y * Block_Width + x) - 1);

        // no need to recreate the entire block
        if (bitmap & bit) {
            auto offset = __popcll(bitmap - (bitmap&(bit+bit-1)));
            auto value = mat.values[start + offset];
            auto mult = vecX[block_col + x];

            // this looks like in can be done using warp reduction
            atomicAdd(&partial_sum[y], value * mult);
        }
    }

    __syncthreads();

    if (threadIdx.x == 0) {
        atomicAdd(&vecY[block_row * Block_Width + y], partial_sum[y]);
    }
}

// only works with 8x8 blocks
template <typename value_type, size_t Block_Width>
__global__ void par_bl_spmv_kernel2(typename BLMatrix<value_type, Block_Width>::DeviceStruct mat,
                                 const value_type * __restrict__ vecX,
                                 value_type *vecY)
{
    // 2d grid
    auto block_row = blockIdx.y;
    auto block_offset = blockIdx.x;

    // coordinates inside the block
    auto y = threadIdx.y;
    auto x = threadIdx.x;

    auto total = 0;

    for (auto j = mat.bl_row_pointers[block_row] + block_offset,
              end = mat.bl_row_pointers[block_row + 1]; j < end; j += gridDim.x) {
        auto block_col = mat.bl_col_indices[j] * Block_Width;

        auto bitmap = mat.bl_bitmaps[j];
        auto start = mat.bl_starts[j];
        auto bit = 1ULL << ((Block_Width*Block_Width) - (y * Block_Width + x) - 1);

        auto acc = 0;

        // no need to recreate the entire block
        if (bitmap & bit) {
            auto offset = __popcll(bitmap - (bitmap&(bit+bit-1)));
            auto value = mat.values[start + offset];
            auto mult = vecX[block_col + x];
            acc = value * mult;
        }

        // copy values from other threads in the warp
        for (auto offset = Block_Width/2; offset > 0; offset /= 2) {
            //if (threadIdx.x < offset) {
#if __CUDA_ARCH__ >= 300
            acc += __shfl_down_sync(0xFFFFFFFF, acc, offset, Block_Width);
#else
            acc += __shfl_down(acc, offset, Block_Width);
#endif
            //}
        }

        total += acc;
    }

    if (threadIdx.x == 0) {
        //atomicAdd(&vecY[block_row * Block_Width + y], partial_sum[y]);
        atomicAdd(&vecY[block_row * Block_Width + y], total);
    }
}

/////////////////////////////////
         ///  IDEA  ///
/////////////////////////////////
// Que bloque denso sea un CSR //
/////////////////////////////////

static void initial_kindergarten_test() {
    using Matrix = BLMatrix2<value_type>;

    constexpr uint32_t rows = 6;
    constexpr uint32_t cols = 6;

    std::vector<value_type> values{{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}};
    std::vector<uint32_t> bl_starts{{0, 2, 3, 4, 5, 7, 8, 12}};
    std::vector<uint64_t> bl_bitmaps{{9, 2, 2, 1, 3, 1, 15}};
    std::vector<uint32_t> bl_col_indices{{0, 1, 2, 0, 2, 0, 1}};
    std::vector<uint32_t> bl_row_pointers{{0, 3, 5, 7}};

    std::vector<value_type> vectX{{1, 1, 2, 1, 1, 2}};
    std::vector<value_type> vectY{{0, 0, 0, 0, 0, 0}};
    std::vector<value_type> y_expected{{1, 12, 0, 25, 28, 42}};

    Matrix mat{rows,
               cols,
               values,
               bl_starts,
               bl_bitmaps,
               bl_col_indices,
               bl_row_pointers};

    std::cout << "Matrix:" << std::endl;
    std::cout << mat << std::endl;

    print_vector("vectX", vectX.data(), vectX.size());

    serial_bl_spmv_kernel_host<value_type, 2>(mat, vectX.data(), &vectY[0]);

    print_vector("vectY", vectY.data(), vectY.size());

    auto dMat = mat.to_device();
    auto d_vecX = dev_alloc_fill(vectX.size(), vectX.data());
    auto d_vecY = dev_alloc_zero<value_type>(vectY.size());
    auto d_expected = dev_alloc_fill(y_expected.size(), y_expected.data());

    {
        constexpr dim3 dimGrid{1, rows/Matrix::block_width};
        constexpr dim3 dimBlock{Matrix::block_width, Matrix::block_width};
        par_bl_spmv_kernel1<value_type, Matrix::block_width><<<dimGrid, dimBlock>>>(dMat, d_vecX.get(), d_vecY.get());
        CUDA_CHK(cudaGetLastError());
        CUDA_CHK(cudaDeviceSynchronize());
        validate_results("par_bl_spmv_kernel1", d_expected.get(), d_vecY.get(), rows);
    }

    {
        cudaMemset(d_vecY.get(), 0, vectY.size() * sizeof(value_type));
        constexpr dim3 dimGrid{1, rows/Matrix::block_width};
        constexpr dim3 dimBlock{Matrix::block_width, Matrix::block_width};
        par_bl_spmv_kernel2<value_type, Matrix::block_width><<<dimGrid, dimBlock>>>(dMat, d_vecX.get(), d_vecY.get());
        CUDA_CHK(cudaGetLastError());
        CUDA_CHK(cudaDeviceSynchronize());
        validate_results("par_bl_spmv_kernel2", d_expected.get(), d_vecY.get(), rows);
    }

    mat.device_free(dMat);
}

void ejercicio2() {
    initial_kindergarten_test();

    using Matrix = BLMatrix2<value_type>;
    lab_tests_controller([&](float p, uint32_t rows, uint32_t columns) {
        Matrix mat{rows, columns};
        mat.random_init(p);

        std::vector<value_type> h_vecX(columns, static_cast<value_type>(1));
        std::vector<value_type> h_vecY(rows, static_cast<value_type>(0));

        auto dMat = mat.to_device();
        auto d_vecX = dev_alloc_fill(h_vecX.size(), h_vecX.data());
        auto d_vecY = dev_alloc_zero<value_type>(rows);

        auto run_kernel = [&](std::string name,
                            std::function<void()> f,
                            value_type *d_expectedResult = nullptr,
                            int times = BENCH_TIMES) {
            std::cout << "---------------------\n" << name << std::endl;
            Metric m;
            for (auto i = 0; i < times; ++i) {
                CUDA_CHK(cudaMemset(d_vecY.get(), 0, sizeof(value_type)*rows));
                auto t = m.track_begin();
                f();
                CUDA_CHK(cudaGetLastError());
                CUDA_CHK(cudaDeviceSynchronize());
                m.track_end(t);
                if (d_expectedResult) {
                    if (!validate_results(name.c_str(), d_expectedResult, d_vecY.get(), rows, i>0, false)) {
                        break;
                    }
                }
            }
            printf("> %f ms, stdev: %f, CV: %f\n", m.mean(), m.stdev(), m.cv());
        };

        run_kernel("ser_bl_spmv_kernel_host", [&]() {
            serial_bl_spmv_kernel_host(mat, h_vecX.data(), h_vecY.data());
        }, nullptr, 1);
        // assume result of serial operation as ground truth
        auto d_expectedResult = dev_alloc_fill<value_type>(rows, h_vecY.data());

        std::vector<uint32_t> blocks_per_row{{1}}; //{{1, 2, 4, 8, 16}};

        for (auto bp : blocks_per_row) {
            std::string name = std::string{"par_bl_spmv_kernel1"} + " (" + std::to_string(bp) + " blocks per block-row)";
            run_kernel(name, [&]() {
                dim3 dimGrid{bp, rows/Matrix::block_width};
                dim3 dimBlock{Matrix::block_width, Matrix::block_width};
                par_bl_spmv_kernel1<value_type, Matrix::block_width><<<dimGrid, dimBlock>>>(dMat, d_vecX.get(), d_vecY.get());
            }, d_expectedResult.get());
        }

        for (auto bp : blocks_per_row) {
            std::string name = std::string{"par_bl_spmv_kernel2"} + " (" + std::to_string(bp) + " blocks per block-row)";
            run_kernel(name, [&]() {
                dim3 dimGrid{1, rows/Matrix::block_width};
                dim3 dimBlock{Matrix::block_width, Matrix::block_width};
                par_bl_spmv_kernel2<value_type, Matrix::block_width><<<dimGrid, dimBlock>>>(dMat, d_vecX.get(), d_vecY.get());
            }, d_expectedResult.get());
        }
    });
}
