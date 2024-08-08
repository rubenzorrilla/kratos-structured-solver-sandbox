#pragma once

#include <cmath>
#include <array>
#include <memory>
#include <vector>
#include <chrono>
#include <utility>
#include <fftw3.h>

#include "mesh_utilities.hpp"
#include "pressure_preconditioner.hpp"
#include "pressure_operator.hpp"
#include "operators.hpp"

template<int TDim>
class PressureConjugateGradientSolver
{
public:

    using VectorType = std::vector<double>;

    PressureConjugateGradientSolver() = default;

    PressureConjugateGradientSolver(
        const double AbsTol,
        const double RelTol,
        const unsigned int MaxIter,
        const PressureOperator& rPressureOperator,
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
        
        mrPressureOperator.Apply(rX, aux); //TODO: We can do a mult and add to make this more efficient
        for (unsigned int i = 0; i < mProblemSize; ++i) {
            r_k[i] = rB[i] - aux[i];
        }

        // Check initial residual
        const double res_norm = ComputeResidualNorm(r_k);
        if (res_norm < mAbsTol) {
            mIsConverged = true;
            return mIsConverged;
        } else {
            mIters = 1;
        }

        // Allocate required arrays
        VectorType d_k(mProblemSize); // Current iteration direction
        VectorType d_k_1(mProblemSize); // Next iteration direction
        VectorType r_k_1(mProblemSize); // Next iteration residual
        VectorType z_k(mProblemSize); // Current iteration preconditioner residual projection
        VectorType z_k_1(mProblemSize); // Next iteration preconditioner residual projection

        ApplyPreconditioner(r_k, z_k);
        d_k = z_k;
        // d_k = r_k; // Identity preconditioner

        auto times = std::vector<std::chrono::microseconds>(6);
        auto counter_beg = std::chrono::high_resolution_clock::now();

        while (!mIsConverged) {
            // Compute current iteration residual and solution
            double aux_1 = 0.0;
            double aux_2 = 0.0;

            counter_beg = std::chrono::high_resolution_clock::now();
            mrPressureOperator.Apply(d_k, aux);
            times[0] += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - counter_beg);

            counter_beg = std::chrono::high_resolution_clock::now();
            for (unsigned int i = 0; i < mProblemSize; ++i) {
                // aux_1 += r_k[i] * r_k[i]; // Identity preconditioner
                aux_1 += r_k[i] * z_k[i];
                aux_2 += d_k[i] * aux[i];
            }
            times[1] += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - counter_beg);

            const double alpha_k = aux_1 / aux_2;

            std::cout << "BFR - Iteration " << alpha_k << " " << mIters << " " << r_k_1[0] << " " << r_k[0] <<  std::endl;

            counter_beg = std::chrono::high_resolution_clock::now();
            for (unsigned int i = 0; i < mProblemSize; ++i) {
                rX[i] = rX[i] + alpha_k * d_k[i];
                r_k_1[i] = r_k[i] - alpha_k * aux[i];
            }
            times[2] += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - counter_beg);

            // Check convergence
            double res_norm;
            double res_inc_norm;

            std::cout << "GFR - Iteration " << mIters << " " << r_k_1[0] << " = " << r_k[0] << " - " << alpha_k << " * " << aux[0] <<  std::endl;

            std::tie(res_norm, res_inc_norm) = ComputeResidualNorms(r_k, r_k_1);
            
            std::cout << "Iteration " << mIters << " Res. norm " << res_norm << " Res. inc. norm " << res_inc_norm << std::endl;
            abort();

            if (res_norm < mAbsTol || res_inc_norm / res_norm < mRelTol) {
                mIsConverged = true;
                break;
            } else {
                if (mIters == mMaxIter) {
                    std::cout << "Maximum iterations reached!" << std::endl;
                    break;
                }
            }

            // Update search direction
            double aux_3 = 0.0;
            double aux_4 = 0.0;

            counter_beg = std::chrono::high_resolution_clock::now();
            ApplyPreconditioner(r_k_1, z_k_1);
            times[3] += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - counter_beg);

            counter_beg = std::chrono::high_resolution_clock::now();
            for (unsigned int i = 0; i < mProblemSize; ++i) {
                aux_3 += r_k_1[i] * z_k_1[i];
                aux_4 += r_k[i] * z_k[i];
                // aux_3 += r_k_1[i] * r_k_1[i]; // Identity preconditioner
                // aux_4 += r_k[i] * r_k[i]; // Identity preconditioner
            }
            times[4] += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - counter_beg);
            const double beta_k = aux_3 / aux_4;

            counter_beg = std::chrono::high_resolution_clock::now();
            for (unsigned int i = 0; i < mProblemSize; ++i) {
                d_k_1[i] = z_k_1[i] + beta_k * d_k[i];
                // d_k_1[i] = r_k_1[i] + beta_k * d_k[i]; // Identity preconditioner
            }
            times[5] += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - counter_beg);

            // Update variables for next step
            mIters++;
            d_k = d_k_1;
            r_k = r_k_1;
            z_k = z_k_1;
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

    const PressureOperator& mrPressureOperator;

    const std::shared_ptr<PressurePreconditioner<TDim>>& mpPressurePreconditioner;

    // const VectorType& mrFFTc;

    double ComputeResidualNorm(const VectorType& rRes)
    {
        double res_norm = 0.0;
        for (unsigned int i = 0; i < mProblemSize; ++i) {
            res_norm += rRes[i] * rRes[i];
        }
        return std::sqrt(res_norm);
    }

    std::pair<double, double> ComputeResidualNorms(
        const VectorType& rOldRes,
        const VectorType& rRes)
    {
        double res_norm = 0.0;
        double res_inc_norm = 0.0;
        for (unsigned int i = 0; i < mProblemSize; ++i) {
            res_norm += rRes[i] * rRes[i];
            res_inc_norm += std::pow(rRes[i] - rOldRes[i], 2);
        }
        return std::make_pair(std::sqrt(res_norm), std::sqrt(res_inc_norm));
    }

    void ApplyPreconditioner(
        const VectorType& rInput,
        VectorType& rOutput)
    {
        mpPressurePreconditioner->Apply(rInput, rOutput);

        // std::vector<std::complex<double>> fft_b(mProblemSize);     // Complex array for FFT(x) output
        // std::vector<std::complex<double>> b_complex(mProblemSize); // Complex array for FFT(x) input
        // for (unsigned int i = 0; i < mProblemSize; ++i) {
        //     b_complex[i].real(rInput[i]);
        // }

        // Eigen::FFT<double> fft;
        // std::vector<std::complex<double>> sol(mProblemSize);
        // fft.fwd(fft_b, b_complex);
        // for (unsigned int i = 0; i < mProblemSize; ++i) {
        //     fft_b[i] = fft_b[i] / mrFFTc[i];
        // }
        // fft.inv(sol, fft_b);
        // for (unsigned int i = 0; i < mProblemSize; ++i) {
        //     rOutput[i] = sol[i].real();
        // }

        //TODO: We should use the FFT for real numbers in here

        // fftw_complex *fft_b;
        // fftw_complex *b_complex;
        // fft_b = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * mProblemSize);
        // b_complex = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * mProblemSize);

        // fftw_plan p_b;
        // p_b = fftw_plan_dft(TDim, mrPressureOperator.GetBoxDivisions().data(), b_complex, fft_b, FFTW_FORWARD, FFTW_ESTIMATE);

        // for (unsigned int i = 0; i < mProblemSize; ++i) {
        //     b_complex[i][0] = rInput[i]; // Setting real part
        //     b_complex[i][1] = 0.0; // Setting imaginary part
        // }

        // fftw_execute(p_b);

        // fftw_complex *ifft_aux;
        // fftw_complex *aux_complex;
        // ifft_aux = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * mProblemSize);
        // aux_complex = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * mProblemSize);

        // fftw_plan p_aux;
        // p_aux = fftw_plan_dft(TDim, mrPressureOperator.GetBoxDivisions().data(), aux_complex, ifft_aux, FFTW_BACKWARD, FFTW_ESTIMATE);

        // for (unsigned int i = 0; i < mProblemSize; ++i) {
        //     const double num_real = fft_b[i][0] * mrFFTc[i]; // Note that in here we are assuming that the imaginary part of mrFFTc is zero
        //     const double num_imag = fft_b[i][1] * mrFFTc[i]; // Note that in here we are assuming that the imaginary part of mrFFTc is zero
        //     const double den = std::pow(mrFFTc[i], 2); // Note that in here we are assuming that the imaginary part of mrFFTc is zero
        //     aux_complex[i][0] = num_real / den;
        //     aux_complex[i][1] = num_imag / den;
        // }

        // fftw_execute(p_aux);

        // // Set the output as the normalized IFFT
        // // Note from FFTW documentation "FFTW computes an unnormalized DFT.Thus, computing a forward
        // // followed by a backward transform (or vice versa) results in the original array scaled by n."
        // for (unsigned int i = 0; i < mProblemSize; ++i) {
        //     rOutput[i] = ifft_aux[i][0] / mProblemSize;
        // }

        // fftw_destroy_plan(p_b);
        // fftw_destroy_plan(p_aux);

        // //TODO: Allocate/free this once
        // fftw_free(fft_b);
        // fftw_free(b_complex);
        // fftw_free(ifft_aux);
        // fftw_free(aux_complex);
    }

};