#include <array>
#include <utility>
#include <vector>
#include <Eigen/Dense>

#pragma once

template<int TDim>
class SbmUtilities
{
public:

    static void UpdateSurrogateBoundaryDirichletValues(
        const double MassFactor,
        const std::array<int, TDim> &rBoxDivisions,
        const std::array<double, TDim> &rCellSize,
        const Eigen::Array<bool, Eigen::Dynamic, 1> &rSurrogateCells,
        const Eigen::Array<bool, Eigen::Dynamic, 1> &rSurrogateNodes,
        const Eigen::Array<double, Eigen::Dynamic, TDim> &rDistanceVects,
        const Eigen::Array<double, Eigen::Dynamic, TDim> &rLumpedMassVector,
        const Eigen::Array<double, Eigen::Dynamic, TDim> &rVelocity,
        Eigen::Array<double, Eigen::Dynamic, TDim> &rSurrogateVelocity);

};