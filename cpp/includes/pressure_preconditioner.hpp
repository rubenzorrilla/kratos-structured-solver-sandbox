#include <array>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <unsupported/Eigen/FFT>

#include "mesh_utilities.hpp"
#include "operators.hpp"

#pragma once

class PressurePreconditioner
{
public:

    explicit PressurePreconditioner(const Eigen::VectorXd& rFFTc)
    : mIsInitialized(true)
    , mpFFTc(&rFFTc)
    {
    }

    template<typename MatrixType>
    PressurePreconditioner& analyzePattern(const MatrixType& )
    {
        return *this;
    }

    template<typename MatrixType>
    PressurePreconditioner& factorize(const MatrixType& )
    {
        return *this;
    }

    template<typename MatrixType>
    PressurePreconditioner& compute(const MatrixType& )
    {
        return *this;
    }

    template<typename Rhs>
    inline const Rhs& solve(const Rhs& b) const
    {
        //TODO: Avoid these copies
        const unsigned int num_cells = mpFFTc->size();
        Eigen::Matrix<std::complex<double>, Eigen::Dynamic, 1> fft_b(num_cells); // Complex array for FFT(x) output
        Eigen::Matrix<std::complex<double>, Eigen::Dynamic, 1> b_complex(num_cells, 1); // Complex array for FFT(x) input
        for (unsigned int i = 0; i < num_cells; ++i) {
            b_complex(i) = b(i);
        }

        Eigen::FFT<double> fft;
        fft.fwd(fft_b, b_complex);
        fft_b.cwiseQuotient(*mpFFTc);
        b = (fft.inv(fft_b)).real();

        return b;
    }

    Eigen::ComputationInfo info()
    {
        return Eigen::Success;
    }

private:

    bool mIsInitialized = false;

    const Eigen::VectorXd* mpFFTc = nullptr;

};
