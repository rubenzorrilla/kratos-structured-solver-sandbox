#include <array>
#include <utility>
#include <vector>
#include <Eigen/Dense>

#include "include/experimental/mdspan"

#pragma once

template<int TDim>
class MeshUtilities
{
public:

    using ExtentsType = std::experimental::extents<std::size_t, std::dynamic_extent, TDim>;

    using MatrixViewType = std::experimental::mdspan<double, ExtentsType>;

    static std::pair<unsigned int, unsigned int> CalculateMeshData(const std::array<int, TDim>& rBoxDivisions);

    static std::array<double, TDim> CalculateCellSize(
        const std::array<double, TDim>& rBoxSize,
        const std::array<int, TDim>& rBoxDivisions);

    static void CalculateNodalCoordinates(
        const std::array<double, TDim>& rBoxSize,
        const std::array<int, TDim>& rBoxDivisions,
        MatrixViewType& rNodalCoords);

    static void CalculateLumpedMassVector(
        const double MassFactor,
        const std::array<int, TDim>& rBoxDivisions,
        const std::vector<bool>& rActiveCells,
        Eigen::Array<double, Eigen::Dynamic, TDim>& rLumpedMassVector);

    static std::tuple<bool, unsigned int> FindFirstFreeCellId(
        const std::array<int, TDim> &rBoxDivisions,
        const Eigen::Array<bool, Eigen::Dynamic, TDim>& rFixity);
};