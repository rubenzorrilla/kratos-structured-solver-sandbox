#include <array>
#include <fstream>
#include <iomanip>

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

    /**
     * @brief Applies the pressure operator to the input vector
     * This function applies the D*inv(LumpedMass)*G pressure operator to the provided vector.
     * Note that cell activation is considered depending on whether the active cells array
     * has been provided or not in the constructor. This allows to compute the circulant
     * of the pressure operator for the FFT preconditioner.
     * @param rInput Input vector to which the pressure operator is applied
     * @param rOutput Output vector
     */
    void Apply(
        const VectorType& rInput,
        VectorType& rOutput) const
    {
        // Allocate auxiliary array
        //TODO: This must be done once
        const unsigned int n_nodes = std::get<0>(MeshUtilities<TDim>::CalculateMeshData(mrBoxDivisions));
        double * aux_data = (double *)malloc(sizeof(double) * n_nodes * TDim);
        MatrixViewType aux(aux_data, n_nodes, TDim);

        // Apply gradient operator to input vector
        Operators<TDim>::ApplyGradientOperator(mrBoxDivisions, mrCellSize, mrActiveCells, rInput, aux);

        // Apply the lumped mass inverse to the auxiliary vector
        for (unsigned int i = 0; i < mrLumpedMassVectorInv.extent(0); ++i) {
            for (unsigned int j = 0; j < mrLumpedMassVectorInv.extent(1); ++j) {
                aux(i, j) *= mrLumpedMassVectorInv(i, j);
            }
        }

        // Apply the divergence operator to the auxiliary vector and store the result in the output array
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

        // // if (mpActiveCells != nullptr) {
        //     const double tau = 1.0e-1;
        //     for (unsigned int i = 0; i < mrBoxDivisions[1]; ++i) {
        //         for (unsigned int j = 0; j < mrBoxDivisions[0]; ++j) {
        //             double temp = 0.0;
        //             unsigned int cell_id = CellUtilities::GetCellGlobalId(i, j, mrBoxDivisions);
        //             // if (GetActiveCells()[cell_id]) {
        //                 // Bottom cell
        //                 if (i > 0) {
        //                     unsigned int neigh_cell_id = CellUtilities::GetCellGlobalId(i - 1, j, mrBoxDivisions);
        //                     // if (GetActiveCells()[cell_id]) {
        //                         temp += mrCellSize[0] * (rInput[cell_id] - rInput[neigh_cell_id]);
        //                     // }
        //                 }
        //                 // Top cell
        //                 if (i < mrBoxDivisions[1] - 1) {
        //                     unsigned int neigh_cell_id = CellUtilities::GetCellGlobalId(i + 1, j, mrBoxDivisions);
        //                     // if (GetActiveCells()[cell_id]) {
        //                         temp += mrCellSize[0] * (rInput[cell_id] - rInput[neigh_cell_id]);
        //                     // }
        //                 }
        //                 // Left cell
        //                 if (j > 0) {
        //                     unsigned int neigh_cell_id = CellUtilities::GetCellGlobalId(i, j - 1, mrBoxDivisions);
        //                     // if (GetActiveCells()[cell_id]) {
        //                         temp += mrCellSize[1] * (rInput[cell_id] - rInput[neigh_cell_id]);
        //                     // }
        //                 }
        //                 // Right cell
        //                 if (j < mrBoxDivisions[0] - 1) {
        //                     unsigned int neigh_cell_id = CellUtilities::GetCellGlobalId(i, j + 1, mrBoxDivisions);
        //                     // if (GetActiveCells()[cell_id]) {
        //                         temp += mrCellSize[1] * (rInput[cell_id] - rInput[neigh_cell_id]);
        //                     // }
        //                 }
        //                 rOutput[cell_id] += tau * temp;
        //             // }
        //         }
        //     }
        // // }

        free(aux_data);
    }

    void Output(
        const std::string Filename,
        const std::string OutputPath = "") const
    {
        // Allocate auxiliary data
        const unsigned int n_cells = std::get<1>(MeshUtilities<TDim>::CalculateMeshData(mrBoxDivisions));
        std::vector<double> aux_vect(n_cells);
        std::vector<std::vector<double>> pressure_operator(n_cells);

        // Obtain the columns of the pressure operator
        for (unsigned int i = 0; i < n_cells; ++i) {
            // Initialize output column
            auto& r_out_col = pressure_operator[i];
            r_out_col.resize(n_cells, 0.0);

            // Initialize current auxiliary column
            std::fill(aux_vect.begin(), aux_vect.end(), 0.0);
            aux_vect[i] = 1.0;

            // Apply the corresponding pressure operator to get current column
            Apply(aux_vect, r_out_col);
        }

        // Get the non-zero entries
        const double tol = 1.0e-14;
        std::vector<std::tuple<unsigned int, unsigned int, double>> non_zero_entries;
        for (unsigned int j = 0; j < n_cells; ++j) {
            const auto& r_j_col = pressure_operator[j];
            for (unsigned int i = 0; i < n_cells; ++i) {
                const double val = r_j_col[i];
                if (std::abs(val) > tol) {
                    non_zero_entries.push_back(std::make_tuple(i, j, val));
                }
            }
        }

        // Output the non-zero entries in (plain) matrix market format
        std::ofstream out_file(OutputPath + Filename + ".mm");
        if (out_file.is_open()) {
            out_file <<  std::scientific << std::setprecision(12);
            out_file << "%%MatrixMarket matrix coordinate real general" << std::endl;
            out_file << n_cells << "  " << n_cells << "  " << non_zero_entries.size() << std::endl;
            for (auto& r_non_zero_entry : non_zero_entries) {
                const unsigned int row = std::get<0>(r_non_zero_entry);
                const unsigned int col = std::get<1>(r_non_zero_entry);
                const double val = std::get<2>(r_non_zero_entry);
                out_file << row + 1 << "  " << col + 1 << "  " << val << std::endl;
            }
            out_file.close();
        }
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