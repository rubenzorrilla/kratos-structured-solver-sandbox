#include <array>
#include <utility>
#include <vector>
#include <Eigen/Dense>

#pragma once

template<int TDim>
class SbmUtilities
{
public:

    static constexpr std::size_t NumNodes = TDim == 2 ? 4: 8;

    using ExtentsType = std::experimental::extents<std::size_t, std::dynamic_extent, TDim>;

    using MatrixViewType = std::experimental::mdspan<double, ExtentsType>;

    using NodesDimExtentsType = std::experimental::extents<std::size_t, NumNodes, TDim>;

    using NodesDimViewType = std::experimental::mdspan<double, NodesDimExtentsType>;

    using DimDimExtentsType = std::experimental::extents<std::size_t, TDim, TDim>;

    using DimDimViewType = std::experimental::mdspan<double, DimDimExtentsType>;

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
        const Eigen::Array<double, Eigen::Dynamic, TDim> &rLumpedMassVector,
        const MatrixViewType &rDistanceVects,
        const MatrixViewType &rVelocity,
        MatrixViewType &rSurrogateVelocity);

};