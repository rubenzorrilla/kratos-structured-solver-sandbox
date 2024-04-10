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

    using VectorType = Eigen::Array<double, Eigen::Dynamic, 1>;

    PressureConjugateGradientSolver() = default;

    PressureConjugateGradientSolver(
        const double AbsTol,
        const unsigned int MaxIter,
        const PressureOperator<TDim>& rPressureOperator,
        const VectorType& rFFTc)
        : mAbsTol(AbsTol)
        , mMaxIter(MaxIter)
        , mrPressureOperator(rPressureOperator)
        , mrFFTc(rFFTc)
    {
        if (!mrPressureOperator.IsInitialized()) {
            std::cerr << "Provided pressure operator is not initialized." << std::endl;
        }
        mProblemSize = mrPressureOperator.ProblemSize();
    }

    void Solve(
        const VectorType& rRHS,
        VectorType& rX)
    {
        std::cout << rRHS << std::endl;
        std::cout << rX << std::endl;

        VectorType aux(mProblemSize);
        mrPressureOperator.Apply(rX, aux);

        VectorType r_k(mProblemSize);
        VectorType r_k_1(mProblemSize);
        r_k = rRHS - aux;

        VectorType d_k(mProblemSize);
        VectorType p_r_k(mProblemSize);
        VectorType d_k_1(mProblemSize);
        VectorType p_r_k_1(mProblemSize);

        std::cout << r_k << std::endl;
        std::cout << p_r_k << std::endl;

        ApplyPreconditioner(r_k, p_r_k);
        d_k = p_r_k;

        unsigned int k = 0;
        while (k < mMaxIter) {
            const double res_norm = ComputeResidualNorm(r_k);
            if (res_norm < mAbsTol) {
                break;
            }

            std::cout << "Iteration " << k << " - Abs. res norm " << res_norm << std::endl;

            mrPressureOperator.Apply(d_k, aux);
            double aux_1 = 0.0;
            double aux_2 = 0.0;
            for (unsigned int i = 0; i < mProblemSize; ++i) {
                aux_1 += r_k(i) * p_r_k(i);
                aux_2 += d_k(i) * aux(i);
            }
            const double alpha = aux_1 / aux_2;

            for (unsigned int i = 0; i < mProblemSize; ++i) {
                rX(i) += alpha * d_k(i);
                r_k_1(i) = r_k(i) - alpha * aux(i);
            }

            ApplyPreconditioner(r_k, p_r_k);
            ApplyPreconditioner(r_k_1, p_r_k_1);

            double aux_3 = 0.0;
            double aux_4 = 0.0;
            for (unsigned int i = 0; i < mProblemSize; ++i) {
                aux_3 += r_k_1(i) * p_r_k_1(i);
                aux_4 += r_k(i) * p_r_k(i);
            }
            const double beta = aux_3 / aux_4;

            for (unsigned int i = 0; i < mProblemSize; ++i) {
                d_k_1(i) = p_r_k_1(i) + beta * d_k(i);
            }

            d_k = d_k_1;
            r_k = r_k_1;
            k++;
        }
    }

private:

    double mAbsTol;

    unsigned int mMaxIter;

    unsigned int mProblemSize;

    const PressureOperator<TDim>& mrPressureOperator;

    const VectorType& mrFFTc;

    double ComputeResidualNorm(const VectorType& rRes)
    {
        double res_norm = 0.0;
        for (unsigned int i = 0; i < mProblemSize; ++i) {
            res_norm += rRes(i) * rRes(i);
        }
        return std::sqrt(res_norm);
    }

    void ApplyPreconditioner(
        const VectorType& rInput,
        VectorType& rOutput)
    {
        Eigen::Matrix<std::complex<double>, Eigen::Dynamic, 1> fft_b(mProblemSize);     // Complex array for FFT(x) output
        Eigen::Matrix<std::complex<double>, Eigen::Dynamic, 1> b_complex(mProblemSize); // Complex array for FFT(x) input
        for (unsigned int i = 0; i < mProblemSize; ++i) {
            b_complex(i) = rInput(i);
        }

        Eigen::FFT<double> fft;
        Eigen::Matrix<std::complex<double>, Eigen::Dynamic, 1> sol(mProblemSize);
        fft.fwd(fft_b, b_complex);
        for (unsigned int i = 0; i < mProblemSize; ++i) {
            fft_b(i) = fft_b(i) / mrFFTc(i);
        }
        fft.inv(sol, fft_b);
        rOutput = sol.real();
    }

};