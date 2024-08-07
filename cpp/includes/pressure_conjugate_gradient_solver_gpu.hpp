#pragma once

#include <cmath>
#include <array>
#include <memory>
#include <vector>
#include <chrono>
#include <utility>
#include <fftw3.h>

// Intel sycl
#include <CL/sycl.hpp>

#include "mesh_utilities.hpp"
#include "pressure_preconditioner.hpp"
#include "pressure_operator.hpp"
#include "operators.hpp"

template<int TDim>
class PressureConjugateGradientSolverGPU
{
public:

    using VectorType = std::vector<double>;

    PressureConjugateGradientSolverGPU() = default;

    PressureConjugateGradientSolverGPU(
        const double AbsTol,
        const double RelTol,
        const unsigned int MaxIter,
        PressureOperator& rPressureOperator,
        const std::shared_ptr<PressurePreconditioner<TDim>>& rpPressurePreconditioner)
        : mAbsTol(AbsTol)
        , mRelTol(RelTol)
        , mMaxIter(MaxIter)
        , mrPressureOperator(rPressureOperator)
        , mpPressurePreconditioner(rpPressurePreconditioner)
    {
        if (!mrPressureOperator.IsInitialized()) {
            std::cerr << "Provided pressure operator is not initialized." << std::endl;
        }
        mProblemSize = mrPressureOperator.ProblemSize();
        mLocalProblemSize = 512;
    }

    const unsigned int Iterations() const
    {
        return mIters;
    }

    const unsigned int IsConverged() const
    {
        return mIsConverged;
    }

    bool Solve(VectorType& rB, VectorType& rX)
    {
        // Initialize data
        mIters = 0;
        mIsConverged = false;

        // Compute the problem size to fit in the gpu and be divisible by the local problem size
        const std::size_t gpu_frame_size = (std::size_t(mProblemSize / mLocalProblemSize) + 1) * mLocalProblemSize;
        std::cout << "mProblemSize: " << mProblemSize << " scaling to: " << gpu_frame_size << std::endl;

        if (rB.size() < gpu_frame_size) rB.resize(gpu_frame_size);
        if (rX.size() < gpu_frame_size) rX.resize(gpu_frame_size);

        // Compute initial residual
        VectorType r_k(gpu_frame_size, 0);
        VectorType aux(gpu_frame_size, 0);

        VectorType d_k(gpu_frame_size, 0);
        VectorType d_k_1(gpu_frame_size, 0);
        VectorType r_k_1(gpu_frame_size, 0);
        VectorType z_k(gpu_frame_size, 0);
        VectorType z_k_1(gpu_frame_size, 0);

        auto times = std::vector<std::chrono::microseconds>(6);
        auto counter_beg = std::chrono::high_resolution_clock::now();

        double d_sum = 0.0;

        // GPU implementation of the gradient operator
        {
            cl::sycl::queue queue(cl::sycl::default_selector_v);
            // cl::sycl::queue queue(cl::sycl::cpu_selector_v);
            // cl::sycl::queue queue(cl::sycl::gpu_selector_v);
            // cl::sycl::queue queue(cl::sycl::accelerator_selector_v);

            mrPressureOperator.Apply(rX, aux); //TODO: We can do a mult and add to make this more efficient
            for (unsigned int i = 0; i < gpu_frame_size; ++i) {
                r_k[i] = rB[i] - aux[i];
            }

            auto resultBuf = sycl::buffer<double>(&d_sum, 1);
            
            auto aux_gpu_buff   = sycl::buffer{aux.data(),      cl::sycl::range<1>{gpu_frame_size}};
            auto r_k_gpu_buff   = sycl::buffer{r_k.data(),      cl::sycl::range<1>{gpu_frame_size}};

            // auto rB_gpu_buff    = sycl::buffer{rB.data(),       cl::sycl::range<1>{rB.size()}};
            // auto rX_gpu_buff    = sycl::buffer{rX.data(),       cl::sycl::range<1>{rX.size()}};

            // auto d_k_gpu_buff   = sycl::buffer{d_k.data(),      cl::sycl::range<1>{gpu_frame_size}};
            // auto d_k_1_gpu_buff = sycl::buffer{d_k_1.data(),    cl::sycl::range<1>{gpu_frame_size}};
            // auto r_k_1_gpu_buff = sycl::buffer{r_k_1.data(),    cl::sycl::range<1>{gpu_frame_size}};
            // auto z_k_gpu_buff   = sycl::buffer{z_k.data(),      cl::sycl::range<1>{gpu_frame_size}};
            // auto z_k_1_gpu_buff = sycl::buffer{z_k_1.data(),    cl::sycl::range<1>{gpu_frame_size}};
            
            // queue.submit([&](cl::sycl::handler & cgh) {
            //     cl::sycl::accessor aux_gpu_acc{aux_gpu_buff, cgh, sycl::read_write};
            //     cl::sycl::accessor r_k_gpu_acc{r_k_gpu_buff, cgh, sycl::read_write};
            //     cl::sycl::accessor rB_gpu_acc{rB_gpu_buff, cgh, sycl::read_only};
            //     cl::sycl::accessor rX_gpu_acc{rB_gpu_buff, cgh, sycl::read_write};
                
            //     cgh.parallel_for<class GPU_PressureOperator_Apply>(cl::sycl::range<1>(gpu_frame_size), [=, this](cl::sycl::item<1> item) {
            //         auto idx = item.get_id(0);
                    
            //         mrPressureOperator.ApplyGPU(rX_gpu_acc, aux_gpu_acc);

            //         r_k_gpu_acc[idx] = rB_gpu_acc[idx] - aux_gpu_acc[idx];
            //     });

            //     queue.wait();
            // });

            double res_norm;

            // Check initial residual
            queue.submit([&aux_gpu_buff, &r_k_gpu_buff, &queue, &resultBuf, &gpu_frame_size, lLocalProbemSize=this->mLocalProblemSize](cl::sycl::handler & cgh) {
                cl::sycl::accessor aux_gpu_acc{aux_gpu_buff, cgh, sycl::read_write};
                cl::sycl::accessor r_k_gpu_acc{r_k_gpu_buff, cgh, sycl::read_write};

                cgh.parallel_for(cl::sycl::nd_range(
                        sycl::range<1>{static_cast<int>(gpu_frame_size)},     // Global size
                        sycl::range<1>{static_cast<int>(lLocalProbemSize)}    // Local size
                    ), sycl::reduction(resultBuf, cgh, std::plus<double>()),
                [=](sycl::nd_item<1> it, auto& sum)
                {
                    std::size_t idx  = it.get_global_id(0);
                    std::size_t size = it.get_global_range(0);
                    for (std::size_t i = idx; i < size; i += size) {
                        sum += r_k_gpu_acc[idx] * r_k_gpu_acc[idx];
                    }
                });

                queue.wait();
            });
            
            sycl::host_accessor result {resultBuf, sycl::read_only};
            res_norm = std::sqrt(result[0]);

            const double cpu_res_norm = ComputeResidualNorm(r_k);
            
            std::cout << "Initial residual norm: " << res_norm << " " << cpu_res_norm << std::endl;

            // if (res_norm < mAbsTol) {
            //     mIsConverged = true;
            //     return mIsConverged;
            // } else {
            //     mIters = 1;
            // }

            // ApplyPreconditioner(r_k, z_k);
            
            // d_k = z_k;

            // while (!mIsConverged && mIters < mMaxIter) {
            //     // Compute current iteration residual and solution
            //     double aux_1 = 0.0;
            //     double aux_2 = 0.0;

            //     counter_beg = std::chrono::high_resolution_clock::now();
            //     mrPressureOperator.Apply(d_k, aux);
            //     times[0] += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - counter_beg);

            //     counter_beg = std::chrono::high_resolution_clock::now();
            //     for (unsigned int i = 0; i < mProblemSize; ++i) {
            //         // aux_1 += r_k[i] * r_k[i]; // Identity preconditioner
            //         aux_1 += r_k[i] * z_k[i];
            //         aux_2 += d_k[i] * aux[i];
            //         if(aux[i] > 0.0) {
            //             std::cout << "#: " <<  r_k[i] << " " << z_k[i] << " " << d_k[i] << " " << aux[i] << " " << aux_1 << std::endl;
            //             abort();
            //         }
            //     }
            //     times[1] += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - counter_beg);

            //     std::cout << "AUXES: " << aux_1 << " " << aux_2 << std::endl;

            //     abort();

            //     const double alpha_k = aux_1 / aux_2; 

            //     std::cout << "BFR - Iteration " << alpha_k << " " << mIters << " " << r_k_1[0] << " " << r_k[0] <<  std::endl;

            //     counter_beg = std::chrono::high_resolution_clock::now();
            //     queue.submit([&](cl::sycl::handler & cgh) {
            //         cl::sycl::accessor rX_gpu_acc{rX_gpu_buff, cgh, sycl::read_write};
            //         cl::sycl::accessor r_k_1_gpu_acc{r_k_1_gpu_buff, cgh, sycl::read_write};
 
            //         cl::sycl::accessor aux_gpu_acc{aux_gpu_buff, cgh, sycl::read_only};
            //         cl::sycl::accessor d_k_gpu_acc{d_k_gpu_buff, cgh, sycl::read_only};
            //         cl::sycl::accessor r_k_gpu_acc{r_k_gpu_buff, cgh, sycl::read_only};

            //         cgh.parallel_for(cl::sycl::range<1>(mProblemSize), [=](cl::sycl::item<1> item) {
            //             auto idx = item.get_id(0);
            //             rX_gpu_acc[idx] = rX_gpu_acc[idx] + alpha_k * d_k_gpu_acc[idx];
            //             r_k_1_gpu_acc[idx] = r_k_gpu_acc[idx] - alpha_k * aux_gpu_acc[idx];
            //         });

            //         queue.wait();
            //     });
            //     times[2] += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - counter_beg);

            //     std::cout << "GFR - Iteration " << mIters << " " << r_k_1[0] << " " << r_k[0] <<  std::endl;

            //     // Check convergence
            //     double res_norm;
            //     double res_inc_norm;

            //     std::tie(res_norm, res_inc_norm) = ComputeResidualNorms(r_k, r_k_1);
            //     std::cout << "AFT - Iteration " << mIters << " Res. norm " << res_norm << " Res. inc. norm " << res_inc_norm << " " << r_k_1[0] << " " << r_k[0] <<  std::endl;

            //     if (!mIsConverged && mIters < mMaxIter) 
            //     {
            //         // Update search direction
            //         double aux_3 = 0.0;
            //         double aux_4 = 0.0;

            //         counter_beg = std::chrono::high_resolution_clock::now();
            //         ApplyPreconditioner(r_k_1, z_k_1);
            //         times[3] += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - counter_beg);

            //         counter_beg = std::chrono::high_resolution_clock::now();
            //         for (unsigned int i = 0; i < mProblemSize; ++i) {
            //             aux_3 += r_k_1[i] * z_k_1[i];
            //             aux_4 += r_k[i] * z_k[i];
            //         }
            //         times[4] += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - counter_beg);
                    
            //         const double beta_k = aux_3 / aux_4;

            //         counter_beg = std::chrono::high_resolution_clock::now();
            //         queue.submit([&](cl::sycl::handler & cgh) {
            //             cl::sycl::accessor d_k_1_gpu_acc{d_k_1_gpu_buff, cgh, sycl::read_write};

            //             cl::sycl::accessor d_k_gpu_acc{d_k_gpu_buff, cgh, sycl::read_only};
            //             cl::sycl::accessor z_k_1_gpu_acc{z_k_1_gpu_buff, cgh, sycl::read_only};

            //             cgh.parallel_for(cl::sycl::range<1>(mProblemSize), [=](cl::sycl::item<1> item) {
            //                 auto idx = item.get_id(0);

            //                 d_k_1_gpu_acc[idx] = z_k_1_gpu_acc[idx] + beta_k * d_k_gpu_acc[idx];
            //             });

            //             queue.wait();
            //         });
            //         times[5] += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - counter_beg);

            //         // Update variables for next step
            //         mIters++;

            //         queue.submit([&](cl::sycl::handler & cgh) {
            //             cl::sycl::accessor d_k_gpu_acc{d_k_gpu_buff, cgh, sycl::read_write};
            //             cl::sycl::accessor r_k_gpu_acc{r_k_gpu_buff, cgh, sycl::read_write};
            //             cl::sycl::accessor z_k_gpu_acc{z_k_gpu_buff, cgh, sycl::read_write};

            //             cl::sycl::accessor d_k_1_gpu_acc{d_k_1_gpu_buff, cgh, sycl::read_only};
            //             cl::sycl::accessor r_k_1_gpu_acc{r_k_1_gpu_buff, cgh, sycl::read_only};
            //             cl::sycl::accessor z_k_1_gpu_acc{z_k_1_gpu_buff, cgh, sycl::read_only};

            //             cgh.parallel_for(cl::sycl::range<1>(mProblemSize), [=](cl::sycl::item<1> item) {
            //                 auto idx = item.get_id(0);

            //                 d_k_gpu_acc[idx] = d_k_1_gpu_acc[idx];
            //                 r_k_gpu_acc[idx] = r_k_1_gpu_acc[idx];
            //                 z_k_gpu_acc[idx] = z_k_1_gpu_acc[idx];
            //             });

            //             queue.wait();
            //         });
            //     }
            // }
        }

        std::cout << "\t 0 in: " << times[0] << std::endl;
        std::cout << "\t 1 in: " << times[1] << std::endl;
        std::cout << "\t 2 in: " << times[2] << std::endl;
        std::cout << "\t 3 in: " << times[3] << std::endl;
        std::cout << "\t 4 in: " << times[4] << std::endl;
        std::cout << "\t 5 in: " << times[5] << std::endl;

        return mIsConverged;
    }

private:


    double mAbsTol;

    double mRelTol;

    bool mIsConverged;

    unsigned int mIters;

    unsigned int mMaxIter;

    unsigned int mProblemSize;

    std::size_t mLocalProblemSize;  //  Maximum size of the local problem such that fits in the GPU registers

    PressureOperator& mrPressureOperator;

    const std::shared_ptr<PressurePreconditioner<TDim>>& mpPressurePreconditioner;

    // const VectorType& mrFFTc;

    template<class res_accessor_t>
    double ComputeResidualNorm(const res_accessor_t& rRes)
    {
        double res_norm = 0.0;
        for (unsigned int i = 0; i < mProblemSize; ++i) {
            res_norm += rRes[i] * rRes[i];
        }
        return std::sqrt(res_norm);
    }

    template<class src_accessor_t, class res_accessor_t>
    std::pair<double, double> ComputeResidualNorms(
        const src_accessor_t& rOldRes,
        const res_accessor_t& rRes)
    {
        double res_norm = 0.0;
        double res_inc_norm = 0.0;
        for (unsigned int i = 0; i < mProblemSize; ++i) {
            res_norm += rRes[i] * rRes[i];
            res_inc_norm += std::pow(rRes[i] - rOldRes[i], 2);
        }
        return std::make_pair(std::sqrt(res_norm), std::sqrt(res_inc_norm));
    }

    template<class src_accessor_t, class dst_accessor_t>
    void ApplyPreconditioner(
        const src_accessor_t& rInput,
        dst_accessor_t& rOutput)
    {
        // mpPressurePreconditioner->Apply(rInput, rOutput);
    }

};