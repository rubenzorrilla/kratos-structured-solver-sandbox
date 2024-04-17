#include <array>
#include <memory>
#include <vector>
#include <utility>
#include <Eigen/Dense>
#include <unsupported/Eigen/FFT>

#include "mesh_utilities.hpp"
#include "pressure_operator.hpp"
#include "operators.hpp"

#pragma once

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
        const PressureOperator<TDim>& rPressureOperator,
        const VectorType& rFFTc)
        : mAbsTol(AbsTol)
        , mRelTol(RelTol)
        , mMaxIter(MaxIter)
        , mrPressureOperator(rPressureOperator)
        , mrFFTc(rFFTc)
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

        while (!mIsConverged) {
            // Compute current iteration residual and solution
            double aux_1 = 0.0;
            double aux_2 = 0.0;
            mrPressureOperator.Apply(d_k, aux);
            for (unsigned int i = 0; i < mProblemSize; ++i) {
                // aux_1 += r_k[i] * r_k[i]; // Identity preconditioner
                aux_1 += r_k[i] * z_k[i];
                aux_2 += d_k[i] * aux[i];
            }
            const double alpha_k = aux_1 / aux_2;

            for (unsigned int i = 0; i < mProblemSize; ++i) {
                rX[i] = rX[i] + alpha_k * d_k[i];
                r_k_1[i] = r_k[i] - alpha_k * aux[i];
            }

            // Check convergence
            double res_norm;
            double res_inc_norm;
            std::tie(res_norm, res_inc_norm) = ComputeResidualNorms(r_k, r_k_1);
            // std::cout << "Iteration " << mIters << " Res. norm " << res_norm << " Res. inc. norm " << res_inc_norm << std::endl;
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
            ApplyPreconditioner(r_k_1, z_k_1);
            for (unsigned int i = 0; i < mProblemSize; ++i) {
                aux_3 += r_k_1[i] * z_k_1[i];
                aux_4 += r_k[i] * z_k[i];
                // aux_3 += r_k_1[i] * r_k_1[i]; // Identity preconditioner
                // aux_4 += r_k[i] * r_k[i]; // Identity preconditioner
            }
            const double beta_k = aux_3 / aux_4;

            for (unsigned int i = 0; i < mProblemSize; ++i) {
                d_k_1[i] = z_k_1[i] + beta_k * d_k[i];
                // d_k_1[i] = r_k_1[i] + beta_k * d_k[i]; // Identity preconditioner
            }

            // Update variables for next step
            mIters++;
            d_k = d_k_1;
            r_k = r_k_1;
            z_k = z_k_1;
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

    const PressureOperator<TDim>& mrPressureOperator;

    const VectorType& mrFFTc;

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
        Eigen::Matrix<std::complex<double>, Eigen::Dynamic, 1> fft_b(mProblemSize);     // Complex array for FFT(x) output
        Eigen::Matrix<std::complex<double>, Eigen::Dynamic, 1> b_complex(mProblemSize); // Complex array for FFT(x) input
        for (unsigned int i = 0; i < mProblemSize; ++i) {
            b_complex(i) = rInput[i];
        }

        Eigen::FFT<double> fft;
        Eigen::Matrix<std::complex<double>, Eigen::Dynamic, 1> sol(mProblemSize);
        fft.fwd(fft_b, b_complex);
        for (unsigned int i = 0; i < mProblemSize; ++i) {
            fft_b(i) = fft_b(i) / mrFFTc[i];
        }
        fft.inv(sol, fft_b);
        for (unsigned int i = 0; i < mProblemSize; ++i) {
            rOutput[i] = (sol.real())(i);
        }

        // const auto& r_active_cells = mrPressureOperator.GetActiveCells();
        // for (unsigned int i = 0; i < mProblemSize; ++i) {
        //     if (r_active_cells[i]) {
        //         rOutput[i] = (sol.real())(i);
        //     } else {
        //         rOutput[i] = 0.0;
        //     }
        // }
    }

};