#include <array>

#include "mesh_utilities.hpp"
#include "operators.hpp"

#pragma once

template<int TDim>
class PressureOperator
{
public:

    using VectorType = std::vector<double>;

    using MatrixViewType = Operators<TDim>::MatrixViewType;

    PressureOperator() = default;

    PressureOperator(
        const std::array<int, TDim>& rBoxDivisions,
        const std::array<double, TDim>& rCellSize,
        const std::vector<bool>& rActiveCells,
        const MatrixViewType& rLumpedMassVectorInv)
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
        const unsigned int n_nodes = std::get<0>(MeshUtilities<TDim>::CalculateMeshData(mrBoxDivisions));
        double aux_data[n_nodes * TDim];
        MatrixViewType aux(aux_data, n_nodes, TDim);
        Operators<TDim>::ApplyGradientOperator(mrBoxDivisions, mrCellSize, mrActiveCells, rInput, aux);
        for (unsigned int i = 0; i < mrLumpedMassVectorInv.extent(0); ++i) {
            for (unsigned int j = 0; j < mrLumpedMassVectorInv.extent(1); ++j) {
                aux(i, j) *= mrLumpedMassVectorInv(i, j);
            }
        }
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

    const std::vector<bool>& GetActiveCells() const
    {
        return mrActiveCells;
    }

private:

    bool mIsInitialized = false;

    const std::array<double, TDim>& mrCellSize;

    const std::array<int, TDim>& mrBoxDivisions;

    const std::vector<bool>& mrActiveCells;

    const MatrixViewType& mrLumpedMassVectorInv;

};