#include <array>
#include <Eigen/Dense>

#include "mesh_utilities.hpp"
#include "operators.hpp"

#pragma once

template<int TDim>
class PressureOperator
{
public:

    using VectorType = std::vector<double>;

    using MatrixType = Eigen::Array<double, Eigen::Dynamic, TDim>;

    PressureOperator() = default;

    PressureOperator(
        const std::array<int, TDim>& rBoxDivisions,
        const std::array<double, TDim>& rCellSize,
        const std::vector<bool>& rActiveCells,
        const MatrixType& rLumpedMassVectorInv)
        : mrCellSize(rCellSize)
        , mrBoxDivisions(rBoxDivisions)
        , mrActiveCells(rActiveCells)
        , mrLumpedMassVectorInv(rLumpedMassVectorInv)
    {
        mIsInitialized = true;
    }

    void Apply(
        const VectorType& rInput,
        VectorType& rOutput) const
    {
        Eigen::Matrix<double, Eigen::Dynamic, TDim> aux;
        Operators<TDim>::ApplyGradientOperator(mrBoxDivisions, mrCellSize, mrActiveCells, rInput, aux);
        aux.array() *= mrLumpedMassVectorInv;
        Operators<TDim>::ApplyDivergenceOperator(mrBoxDivisions, mrCellSize, mrActiveCells, aux, rOutput);
    }

    const bool IsInitialized() const
    {
        return mIsInitialized;
    }

    const unsigned int ProblemSize() const
    {
        return std::get<1>(MeshUtilities<TDim>::CalculateMeshData(mrBoxDivisions));
    }

private:

    bool mIsInitialized = false;

    const std::array<double, TDim>& mrCellSize;

    const std::array<int, TDim>& mrBoxDivisions;

    const std::vector<bool>& mrActiveCells;

    const MatrixType& mrLumpedMassVectorInv;

};