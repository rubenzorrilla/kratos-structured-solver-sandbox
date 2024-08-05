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
    }

    const unsigned int Iterations() const
    {
        return mIters;
    }

    const unsigned int IsConverged() const
    {
        return mIsConverged;
    }

    bool Solve(
        const VectorType& rB,
        VectorType& rX)
    {
        // Initialize data
        mIters = 0;
        mIsConverged = false;

        // Compute initial residual
        VectorType r_k(mProblemSize);
        VectorType aux(mProblemSize);

        // GPU implementation of the gradient operator
        {
            cl::sycl::queue queue(cl::sycl::cpu_selector_v);

            auto aux_gpu_buff = sycl::buffer{aux.data(), cl::sycl::range<1>{mProblemSize}};
            auto r_k_gpu_buff = sycl::buffer{r_k.data(), cl::sycl::range<1>{mProblemSize}};
            auto rB_gpu_buff = sycl::buffer{rB.data(),  cl::sycl::range<1>{rB.size()}};
            auto rX_gpu_buff = sycl::buffer{rX.data(),  cl::sycl::range<1>{rX.size()}};

            queue.submit([&](cl::sycl::handler & cgh) {
                cl::sycl::accessor aux_gpu_acc{aux_gpu_buff, cgh, sycl::read_write};
                cl::sycl::accessor r_k_gpu_acc{r_k_gpu_buff, cgh, sycl::read_write};
                cl::sycl::accessor rB_gpu_acc{rB_gpu_buff, cgh, sycl::read_only};
                cl::sycl::accessor rX_gpu_acc{rB_gpu_buff, cgh, sycl::read_write};
                
                cgh.parallel_for<class GPU_Vanilla>(cl::sycl::range<1>(mProblemSize), [=, this](cl::sycl::item<1> item) {
                    auto idx = item.get_id(0);
                    
                    mrPressureOperator.ApplyGPU(rX_gpu_acc, aux_gpu_acc);

                    r_k_gpu_acc[idx] = rB_gpu_acc[idx] - aux_gpu_acc[idx];
                });

                // Check initial residual
                const double res_norm = ComputeResidualNorm(r_k);
                if (res_norm < mAbsTol) {
                    mIsConverged = true;
                    return mIsConverged;
                } else {
                    mIters = 1;
                }

                sycl::local_accessor<double> d_k(mProblemSize, cgh);
                sycl::local_accessor<double> d_k_1(mProblemSize, cgh);
                sycl::local_accessor<double> r_k_1(mProblemSize, cgh);
                sycl::local_accessor<double> z_k(mProblemSize, cgh);
                sycl::local_accessor<double> z_k_1(mProblemSize, cgh);

                ApplyPreconditioner(r_k_gpu_acc, z_k);
                d_k = z_k;

                auto times = std::vector<std::chrono::microseconds>(6);
                auto counter_beg = std::chrono::high_resolution_clock::now();

                while (!mIsConverged) {
                    // // Compute current iteration residual and solution
                    // double aux_1 = 0.0;
                    // double aux_2 = 0.0;

                    // counter_beg = std::chrono::high_resolution_clock::now();
                    // mrPressureOperator.ApplyGPU(d_k, aux_gpu_acc);
                    // times[0] += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - counter_beg);

                    // counter_beg = std::chrono::high_resolution_clock::now();
                    // for (unsigned int i = 0; i < mProblemSize; ++i) {
                    //     // aux_1 += r_k[i] * r_k[i]; // Identity preconditioner
                    //     aux_1 += r_k_gpu_acc[i] * z_k[i];
                    //     aux_2 += d_k[i] * aux_gpu_acc[i];
                    // }

                    // // TODO: This should look limke this:
                    // // sycl::accessor buf_acc(buf, h, sycl::read_only);
                    // // sycl::accessor accum_acc(accum_buf, h, sycl::write_only, sycl::no_init);
                    // // chg.parallel_for(num_work_items, [=](auto index) {
                    // //     size_t glob_id = index[0];
                    // //     int sum = 0;
                    // //     for (size_t i = glob_id; i < data_size; i += num_work_items) {
                    // //         sum += buf_acc[i];
                    // //     }
                    // //     accum_acc[glob_id] = sum;
                    // // });
                    // times[1] += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - counter_beg);

                //     const double alpha_k = aux_1 / aux_2;

                //     counter_beg = std::chrono::high_resolution_clock::now();
                //     cgh.parallel_for<class GPU_Vanilla>(cl::sycl::range<1>(mProblemSize), [=](cl::sycl::item<1> item) {
                //         auto idx = item.get_id(0);
                //         rX_gpu_acc[idx] = rX_gpu_acc[idx] + alpha_k * d_k[idx];
                //         r_k_1[idx] = r_k_gpu_acc[idx] - alpha_k * aux_gpu_acc[idx];
                //     });
                //     times[2] += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - counter_beg);

                //     // Check convergence
                //     double res_norm;
                //     double res_inc_norm;
                //     std::tie(res_norm, res_inc_norm) = ComputeResidualNorms(r_k_gpu_acc, r_k_1);
                //     // std::cout << "Iteration " << mIters << " Res. norm " << res_norm << " Res. inc. norm " << res_inc_norm << std::endl;
                //     if (res_norm < mAbsTol || res_inc_norm / res_norm < mRelTol) {
                //         mIsConverged = true;
                //         break;
                //     } else {
                //         if (mIters == mMaxIter) {
                //             std::cout << "Maximum iterations reached!" << std::endl;
                //             break;
                //         }
                //     }

                //     // Update search direction
                //     double aux_3 = 0.0;
                //     double aux_4 = 0.0;

                //     counter_beg = std::chrono::high_resolution_clock::now();
                //     ApplyPreconditioner(r_k_1, z_k_1);
                //     times[3] += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - counter_beg);

                //     counter_beg = std::chrono::high_resolution_clock::now();
                //     for (unsigned int i = 0; i < mProblemSize; ++i) {
                //         aux_3 += r_k_1[i] * z_k_1[i];
                //         aux_4 += r_k_gpu_acc[i] * z_k[i];
                //         // aux_3 += r_k_1[i] * r_k_1[i]; // Identity preconditioner
                //         // aux_4 += r_k[i] * r_k[i]; // Identity preconditioner
                //     }
                //     times[4] += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - counter_beg);
                //     const double beta_k = aux_3 / aux_4;

                //     counter_beg = std::chrono::high_resolution_clock::now();
                //     cgh.parallel_for<class GPU_Vanilla>(cl::sycl::range<1>(mProblemSize), [=](cl::sycl::item<1> item) {
                //         auto idx = item.get_id(0);
                //         d_k_1[idx] = z_k_1[idx] + beta_k * d_k[idx];
                //         // d_k_1[idx] = r_k_1[idx] + beta_k * d_k[idx]; // Identity preconditioner
                //     });
                //     times[5] += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - counter_beg);

                //     // Update variables for next step
                //     mIters++;
                //     d_k = d_k_1;
                //     r_k.swap(r_k_1);
                //     z_k = z_k_1;
                }
            });
        }

        return mIsConverged;
    }

private:


    double mAbsTol;

    double mRelTol;

    bool mIsConverged;

    unsigned int mIters;

    unsigned int mMaxIter;

    unsigned int mProblemSize;

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