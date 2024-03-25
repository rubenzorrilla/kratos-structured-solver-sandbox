#include <iostream>

#include "time_utilities.hpp"

template<>
double TimeUtilities<2>::CalculateDeltaTime(
    const double Rho,
    const double Mu,
    const std::array<double, 2>& rCellSize,
    const Eigen::Array<double, Eigen::Dynamic, 2>& rVelocities,
    const double TargetCFL,
    const double TargetFourier)
{
    const double tol = 1.0e-12;
    double max_v = rVelocities.rowwise().norm().maxCoeff();
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
    const Eigen::Array<double, Eigen::Dynamic, 3>& rVelocities,
    const double TargetCFL,
    const double TargetFourier)
{
    const double tol = 1.0e-12;
    double max_v = rVelocities.rowwise().norm().maxCoeff();
    if (max_v < tol) {
        max_v = tol;
    }

    Eigen::Array3d cfl_dt_vect(rCellSize.data());
    cfl_dt_vect *= TargetCFL / max_v;
    const double cfl_dt = cfl_dt_vect.minCoeff();

    Eigen::Array3d fourier_dt_vect(rCellSize.data());
    fourier_dt_vect.pow(2);
    fourier_dt_vect *= Rho * TargetFourier / Mu;
    const double fourier_dt = fourier_dt_vect.minCoeff();

    return std::min(cfl_dt, fourier_dt);
}

template class TimeUtilities<2>;
template class TimeUtilities<3>;
