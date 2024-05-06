#include <array>

#include "cell_utilities.hpp"
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
        // mArtificialCompressibility = false;
    }

    // PressureOperator(
    //     const double Rho,
    //     const double SoundVelocity,
    //     const std::array<int, TDim>& rBoxDivisions,
    //     const std::array<double, TDim>& rCellSize,
    //     const std::vector<bool>& rActiveCells,
    //     const MatrixViewType& rLumpedMassVectorInv)
    //     : mRho(Rho)
    //     , mSoundVelocity(SoundVelocity)
    //     , mrCellSize(rCellSize)
    //     , mrBoxDivisions(rBoxDivisions)
    //     , mrActiveCells(rActiveCells)
    //     , mrLumpedMassVectorInv(rLumpedMassVectorInv)
    // {
    //     mIsInitialized = true;
    //     mArtificialCompressibility = true;
    // }

    // void SetDeltaTime(const double DeltaTime)
    // {
    //     mDeltaTime = DeltaTime;
    // }

    void Apply(
        const VectorType& rInput,
        VectorType& rOutput) const
    {
        // Apply the standard D*Minv*G pressure operator
        // Note that this includes the product by the time increment
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

        // // Append the artificial compressibility contribution
        // if (mArtificialCompressibility) {
        //     const double mass_factor = std::reduce(mrCellSize.begin(), mrCellSize.end(), 1.0, std::multiplies<>());
        //     const double bulk_factor = mass_factor / (mRho * std::pow(mSoundVelocity, 2));
        //     for (unsigned int i = 0; i < rInput.size(); ++i) {
        //         //FIXME: I'm not sure about this sign. This is to be checked!
        //         rOutput[i] += bulk_factor * rInput[i]; //TODO: I think we can use std::transform for this
        //     }
        // }

        // const double tau = 1.0e-3;
        // for (unsigned int i = 0; i < mrBoxDivisions[0]; ++i) {
        //     for (unsigned int j = 0; j < mrBoxDivisions[1]; ++j) {
        //         double temp = 0.0;
        //         unsigned int cell_id = CellUtilities::GetCellGlobalId(i, j, mrBoxDivisions);
        //         if (mrActiveCells[cell_id]) {
        //             if (i > 0) {
        //                 unsigned int neigh_cell_id = CellUtilities::GetCellGlobalId(i - 1, j, mrBoxDivisions);
        //                 if (mrActiveCells[cell_id]) {
        //                     temp += mrCellSize[0] * (rInput[cell_id] - rInput[neigh_cell_id]);
        //                 }
        //             }
        //             if (i < mrBoxDivisions[0] - 1) {
        //                 unsigned int neigh_cell_id = CellUtilities::GetCellGlobalId(i + 1, j, mrBoxDivisions);
        //                 if (mrActiveCells[cell_id]) {
        //                     temp += mrCellSize[0] * (rInput[cell_id] - rInput[neigh_cell_id]);
        //                 }
        //             }
        //             if (j > 0) {
        //                 unsigned int neigh_cell_id = CellUtilities::GetCellGlobalId(i, j - 1, mrBoxDivisions);
        //                 if (mrActiveCells[cell_id]) {
        //                     temp += mrCellSize[1] * (rInput[cell_id] - rInput[neigh_cell_id]);
        //                 }
        //             }
        //             if (i < mrBoxDivisions[1] - 1) {
        //                 unsigned int neigh_cell_id = CellUtilities::GetCellGlobalId(i, j + 1, mrBoxDivisions);
        //                 if (mrActiveCells[cell_id]) {
        //                     temp += mrCellSize[1] * (rInput[cell_id] - rInput[neigh_cell_id]);
        //                 }
        //             }
        //             rOutput[cell_id] -= tau * temp;
        //         }
        //     }
        // }
    }

    const bool IsInitialized() const
    {
        return mIsInitialized;
    }

    const unsigned int ProblemSize() const
    {
        return std::get<1>(MeshUtilities<TDim>::CalculateMeshData(mrBoxDivisions));
    }

    const std::array<int, TDim>& GetBoxDivisions() const
    {
        return mrBoxDivisions;
    }

    const std::vector<bool>& GetActiveCells() const
    {
        return mrActiveCells;
    }

private:

    bool mIsInitialized = false;

    // bool mArtificialCompressibility;

    // double mDeltaTime;

    // const double mRho = 0.0;

    // const double mSoundVelocity = 0.0;

    const std::array<double, TDim>& mrCellSize;

    const std::array<int, TDim>& mrBoxDivisions;

    const std::vector<bool>& mrActiveCells;

    const MatrixViewType& mrLumpedMassVectorInv;

};