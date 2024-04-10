#include <array>
#include <Eigen/Dense>

#pragma once

template<int TDim>
class Operators
{
public:

    //TODO: This is to be eventually removed
    static void ApplyGradientOperator(
        const std::array<int, TDim>& rBoxDivisions,
        const std::array<double, TDim>& rCellSize,
        const Eigen::Array<bool, Eigen::Dynamic, 1>& rActiveCells,
        const Eigen::VectorXd& rX,
        Eigen::Matrix<double, Eigen::Dynamic, TDim>& rOutput);

    //TODO: This is to be eventually removed
    static void ApplyDivergenceOperator(
        const std::array<int, TDim>& rBoxDivisions,
        const std::array<double, TDim>& rCellSize,
        const Eigen::Array<bool, Eigen::Dynamic, 1>& rActiveCells,
        const Eigen::Array<double, Eigen::Dynamic, TDim>& rX,
        Eigen::VectorXd& rOutput);

    static void ApplyGradientOperator(
        const std::array<int, TDim>& rBoxDivisions,
        const std::array<double, TDim>& rCellSize,
        const std::vector<bool>& rActiveCells,
        const Eigen::VectorXd& rX,
        Eigen::Matrix<double, Eigen::Dynamic, TDim>& rOutput);

    static void ApplyDivergenceOperator(
        const std::array<int, TDim>& rBoxDivisions,
        const std::array<double, TDim>& rCellSize,
        const std::vector<bool>& rActiveCells,
        const Eigen::Array<double, Eigen::Dynamic, TDim>& rX,
        Eigen::Array<double, Eigen::Dynamic, 1>& rOutput);

    static void ApplyPressureOperator(
        const std::array<int, TDim>& rBoxDivisions,
        const std::array<double, TDim>& rCellSize,
        const Eigen::Array<bool, Eigen::Dynamic, 1>& rActiveCells,
        const Eigen::Array<double, Eigen::Dynamic, TDim>& rLumpedMassVectorInv,
        const Eigen::VectorXd& rX,
        Eigen::VectorXd& rOutput);
};