#include <array>
#include <utility>
#include <vector>
#include <Eigen/Dense>

#pragma once

template<int TDim>
class SbmUtilities
{
public:

    static void FindSurrogateBoundaryNodes(
        const std::array<int, TDim> &rBoxDivisions,
        const std::vector<double>& rDistance,
        std::vector<bool>& rSurrogateNodes);

    static void FindSurrogateBoundaryCells(
        const std::array<int, TDim> &rBoxDivisions,
        const std::vector<double>& rDistance,
        const std::vector<bool>& rSurrogateNodes,
        std::vector<bool>& rSurrogateCells);

    static void UpdateSurrogateBoundaryDirichletValues(
        const double MassFactor,
        const std::array<int, TDim> &rBoxDivisions,
        const std::array<double, TDim> &rCellSize,
        const std::vector<bool> &rSurrogateCells,
        const std::vector<bool> &rSurrogateNodes,
        const Eigen::Array<double, Eigen::Dynamic, TDim> &rDistanceVects,
        const Eigen::Array<double, Eigen::Dynamic, TDim> &rLumpedMassVector,
        const Eigen::Array<double, Eigen::Dynamic, TDim> &rVelocity,
        Eigen::Array<double, Eigen::Dynamic, TDim> &rSurrogateVelocity);

};