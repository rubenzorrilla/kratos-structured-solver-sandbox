#include <cmath>
#include <iostream>

#include "time_utilities.hpp"

template<>
double TimeUtilities<2>::CalculateDeltaTime(
    const double Rho,
    const double Mu,
    const std::array<double, 2>& rCellSize,
    const MatrixViewType& rVelocities,
    const double TargetCFL,
    const double TargetFourier)
{
    const double tol = 1.0e-12;
    double max_v = TimeUtilities<2>::CalculateMaximumVelocityNorm(rVelocities);
    if (max_v < tol) {
        max_v = tol;
    }

    const double cfl_dt_x = rCellSize[0] * TargetCFL / max_v;
    const double cfl_dt_y = rCellSize[1] * TargetCFL / max_v;
    const double cfl_dt = std::min(cfl_dt_x, cfl_dt_y);

    const double fourier_dt_x = Rho * std::pow(rCellSize[0], 2) * TargetFourier / Mu;
    const double fourier_dt_y = Rho * std::pow(rCellSize[1], 2) * TargetFourier / Mu;
    const double fourier_dt = std::min(fourier_dt_x, fourier_dt_y);

    return std::min(cfl_dt, fourier_dt);
}

template<>
double TimeUtilities<3>::CalculateDeltaTime(
    const double Rho,
    const double Mu,
    const std::array<double, 3>& rCellSize,
    const MatrixViewType& rVelocities,
    const double TargetCFL,
    const double TargetFourier)
{
    const double tol = 1.0e-12;
    double max_v = TimeUtilities<3>::CalculateMaximumVelocityNorm(rVelocities);
    if (max_v < tol) {
        max_v = tol;
    }

    const double cfl_dt_x = rCellSize[0] * TargetCFL / max_v;
    const double cfl_dt_y = rCellSize[1] * TargetCFL / max_v;
    const double cfl_dt_z = rCellSize[2] * TargetCFL / max_v;
    const double cfl_dt = std::min(cfl_dt_x, std::min(cfl_dt_y, cfl_dt_z));

    const double fourier_dt_x = Rho * std::pow(rCellSize[0], 2) * TargetFourier / Mu;
    const double fourier_dt_y = Rho * std::pow(rCellSize[1], 2) * TargetFourier / Mu;
    const double fourier_dt_z = Rho * std::pow(rCellSize[2], 2) * TargetFourier / Mu;
    const double fourier_dt = std::min(fourier_dt_x, std::min(fourier_dt_y, fourier_dt_z));

    return std::min(cfl_dt, fourier_dt);
}

template <int TDim>
double TimeUtilities<TDim>::CalculateMaximumVelocityNorm(const MatrixViewType& rVelocities)
{
    double max_v = 0.0;
    for (unsigned int i = 0; i < rVelocities.extent(0); ++i) {
        double aux_v = 0.0;
        for (unsigned int d = 0; d < TDim; ++d) {
            aux_v += std::pow(rVelocities(i, d), 2);
        }
        if (aux_v > max_v) {
            max_v = aux_v;
        }
    }

    return std::sqrt(max_v);
}

template class TimeUtilities<2>;
template class TimeUtilities<3>;
