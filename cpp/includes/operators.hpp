#include <array>
#include <Eigen/Dense>

#pragma once

template<int TDim>
class Operators
{
public:

    static void ApplyGradientOperator(
        const std::array<int, TDim>& rBoxDivisions,
        const std::array<double, TDim>& rCellSize,
        const std::vector<bool>& rActiveCells,
        const std::vector<double>& rX,
        Eigen::Matrix<double, Eigen::Dynamic, TDim>& rOutput);

    static void ApplyDivergenceOperator(
        const std::array<int, TDim>& rBoxDivisions,
        const std::array<double, TDim>& rCellSize,
        const std::vector<bool>& rActiveCells,
        const Eigen::Array<double, Eigen::Dynamic, TDim>& rX,
        std::vector<double>& rOutput);

    static void ApplyPressureOperator(
        const std::array<int, TDim>& rBoxDivisions,
        const std::array<double, TDim>& rCellSize,
        const std::vector<bool>& rActiveCells,
        const Eigen::Array<double, Eigen::Dynamic, TDim>& rLumpedMassVectorInv,
        const std::vector<double>& rX,
        std::vector<double>& rOutput);
};