#include <array>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <unsupported/Eigen/FFT>

#include "mesh_utilities.hpp"
#include "operators.hpp"

#pragma once

class PressurePreconditioner
{

    typedef Eigen::Matrix<double, Eigen::Dynamic, 1> Vector;

public:

    typedef typename Vector::StorageIndex StorageIndex;
    enum
    {
        ColsAtCompileTime = Eigen::Dynamic,
        MaxColsAtCompileTime = Eigen::Dynamic
    };

    PressurePreconditioner() = default;

    explicit PressurePreconditioner(const Eigen::VectorXd& rFFTc)
    : mIsInitialized(true)
    , mpFFTc(&rFFTc)
    {
    }

    void setFFT(const Eigen::VectorXd& rFFTc)
    {
        mIsInitialized = true;
        mpFFTc = &rFFTc;
    }

    Eigen::Index rows() const
    {
        return mpFFTc->size();
    }

    Eigen::Index cols() const
    {
        return mpFFTc->size();
    }

    template<typename MatrixType>
    PressurePreconditioner& analyzePattern(const MatrixType&)
    {
        return *this;
    }

    template<typename MatrixType>
    PressurePreconditioner& factorize(const MatrixType&)
    {
        return *this;
    }

    template<typename MatrixType>
    PressurePreconditioner& compute(const MatrixType&)
    {
        return *this;
    }

    template<typename Rhs, typename Dest>
    void _solve_impl(const Rhs& b, Dest& x) const
    {
        if (mIsInitialized) {
            //TODO: Avoid these copies
            const unsigned int num_cells = mpFFTc->size();
            Eigen::Matrix<std::complex<double>, Eigen::Dynamic, 1> fft_b(num_cells); // Complex array for FFT(x) output
            Eigen::Matrix<std::complex<double>, Eigen::Dynamic, 1> b_complex(num_cells); // Complex array for FFT(x) input
            for (unsigned int i = 0; i < num_cells; ++i) {
                b_complex(i) = b(i);
            }

            Eigen::FFT<double> fft;
            Eigen::Matrix<std::complex<double>, Eigen::Dynamic, 1> sol(num_cells);
            fft.fwd(fft_b, b_complex);
            fft_b.cwiseQuotient(*mpFFTc);
            fft.inv(sol, fft_b);
            x = sol.real();
        } else {
            std::cout << "Pressure preconditioner is not initialized. Using do-nothing identity one." << std::endl;
        }
    }

    template<typename Rhs>
    inline const Eigen::Solve<PressurePreconditioner, Rhs> solve(const Eigen::MatrixBase<Rhs>& b) const
    {
      return Eigen::Solve<PressurePreconditioner, Rhs>(*this, b.derived());
    }

    Eigen::ComputationInfo info()
    {
        return Eigen::Success;
    }

private:

    bool mIsInitialized = false;

    const Eigen::VectorXd* mpFFTc = nullptr;

};
