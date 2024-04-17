#include <array>
#include <Eigen/Dense>
#include "include/experimental/mdspan"

#pragma once

template<int TDim>
class Operators
{
public:

    using ExtentsType = std::experimental::extents<std::size_t, std::dynamic_extent, TDim>;

    using MatrixViewType = std::experimental::mdspan<double, ExtentsType>;

    static void ApplyGradientOperator(
        const std::array<int, TDim>& rBoxDivisions,
        const std::array<double, TDim>& rCellSize,
        const std::vector<bool>& rActiveCells,
        const std::vector<double>& rX,
        MatrixViewType& rOutput);

    static void ApplyDivergenceOperator(
        const std::array<int, TDim>& rBoxDivisions,
        const std::array<double, TDim>& rCellSize,
        const std::vector<bool>& rActiveCells,
        const MatrixViewType& rX,
        std::vector<double>& rOutput);

    static void ApplyPressureOperator(
        const std::array<int, TDim>& rBoxDivisions,
        const std::array<double, TDim>& rCellSize,
        const std::vector<bool>& rActiveCells,
        const MatrixViewType& rLumpedMassVectorInv,
        const std::vector<double>& rX,
        std::vector<double>& rOutput);
};