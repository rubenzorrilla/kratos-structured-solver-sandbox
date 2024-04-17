#include <array>
#include <utility>
#include <vector>
#include "include/experimental/mdspan"

#pragma once

template<int TDim>
class TimeUtilities
{
public:

    using ExtentsType = std::experimental::extents<std::size_t, std::dynamic_extent, TDim>;

    using MatrixViewType = std::experimental::mdspan<double, ExtentsType>;

    static double CalculateDeltaTime(
        const double Rho,
        const double Mu,
        const std::array<double, TDim>& rCellSize,
        const MatrixViewType& rVelocities,
        const double TargetCFL,
        const double TargetFourier);

private:

    static double CalculateMaximumVelocityNorm(const MatrixViewType& rVelocities);

};