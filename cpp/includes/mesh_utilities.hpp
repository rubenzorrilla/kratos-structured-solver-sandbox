#include <array>
#include <utility>
#include <vector>
#include <Eigen/Dense>

#pragma once

template<int TDim>
class MeshUtilities
{
public:

    static std::pair<unsigned int, unsigned int> CalculateMeshData(const std::array<int, TDim>& rBoxDivisions);

    static std::array<double, TDim> CalculateCellSize(
        const std::array<double, TDim>& rBoxSize,
        const std::array<int, TDim>& rBoxDivisions);

    static void CalculateNodalCoordinates(
        const std::array<double, TDim>& rBoxSize,
        const std::array<int, TDim>& rBoxDivisions,
        Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic>& rNodalCoords);

    static void CalculateLumpedMassVector(
        const double MassFactor,
        const std::array<int, TDim>& rBoxDivisions,
        const Eigen::Array<bool, Eigen::Dynamic, 1>& rActiveCells,
        Eigen::Array<double, Eigen::Dynamic, TDim>& rLumpedMassVector);

    static unsigned int FindFirstFreeCellId(
        const std::array<int, TDim> &rBoxDivisions,
        const Eigen::Array<bool, Eigen::Dynamic, TDim>& rFixity);
};