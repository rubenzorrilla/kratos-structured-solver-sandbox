#include <array>
#include <utility>
#include <vector>
#include <Eigen/Dense>

#pragma once

template<int TDim>
class TimeUtilities
{
public:

    static double CalculateDeltaTime(
        const double Rho,
        const double Mu,
        const std::array<double, TDim>& rCellSize,
        const Eigen::Array<double, Eigen::Dynamic, TDim>& rVelocities,
        const double TargetCFL,
        const double TargetFourier);

};