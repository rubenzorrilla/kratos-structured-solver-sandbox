#include <array>
#include <fstream>
#include <iomanip>

// Intel sycl
#include <CL/sycl.hpp>

#include "cell_utilities.hpp"
#include "mesh_utilities.hpp"
#include "operators.hpp"
#include "incompressible_navier_stokes_q1_p0_structured_element.hpp"

#pragma once

#define TDim 2

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
        , mNnodes(std::get<0>(MeshUtilities<TDim>::CalculateMeshData(mrBoxDivisions)))
    {
        mIsInitialized = true;
        // mArtificialCompressibility = false;
        mAuxData = (double *)malloc(sizeof(double) * mNnodes * TDim);
    }

    ~PressureOperator()
    {
        free(mAuxData);
    }

    // void SetDeltaTime(const double DeltaTime)
    // {
    //     mDeltaTime = DeltaTime;
    // }

    void LocalApplyGradientOperator(
        const std::array<int, TDim>& rBoxDivisions,
        const std::array<double, TDim>& rCellSize,
        const std::vector<bool>& rActiveCells,
        const auto & rX,
        MatrixViewType& rOutput
    ) 
    {
        // Check output matrix sizes (i.e. mdspan extent)
        const unsigned int n_nodes = std::get<0>(MeshUtilities<2>::CalculateMeshData(rBoxDivisions));
        if (rOutput.extent(0) != n_nodes || rOutput.extent(1) != 2) {
            // throw std::logic_error("Wrong size in mdspan extent.");
            // std::cout << "Wrong size in mdspan extent." << std::endl;
        }

        // Initialize output matrix
        for (unsigned int i = 0; i < rOutput.extent(0); ++i) {
            rOutput(i, 0) = 0.0;
            rOutput(i, 1) = 0.0;
        }

        // Get the cell gradient operator
        // Eigen::Array<double,4,2> cell_gradient_operator;
        double cell_gradient_operator_data[8];
        IncompressibleNavierStokesQ1P0StructuredElement::QuadVectorDataView cell_gradient_operator(cell_gradient_operator_data, 4, 2);
        IncompressibleNavierStokesQ1P0StructuredElement::GetCellGradientOperator(rCellSize[0], rCellSize[1], cell_gradient_operator);

        // Apply the gradient operator onto a vector
        std::array<int,4> cell_node_ids;
        for (unsigned int i = 0; i < rBoxDivisions[1]; ++i) {
            for (unsigned int j = 0; j < rBoxDivisions[0]; ++j) {
                const unsigned int cell_id = CellUtilities::GetCellGlobalId(i, j, rBoxDivisions);
                if (rActiveCells[cell_id]) {
                    const double x = rX[cell_id];
                    CellUtilities::GetCellNodesGlobalIds(i, j, rBoxDivisions, cell_node_ids);
                    unsigned int i_node = 0;
                    for (unsigned int id_node : cell_node_ids) {
                        rOutput(id_node, 0) += cell_gradient_operator(i_node, 0) * x;
                        rOutput(id_node, 1) += cell_gradient_operator(i_node, 1) * x;
                        i_node++;
                    }
                }
            }
        }
    }

    void LocalApplyDivergenceOperator(
        const std::array<int, TDim>& rBoxDivisions,
        const std::array<double, TDim>& rCellSize,
        const std::vector<bool>& rActiveCells,
        const MatrixViewType& rX,
        auto & rOutput
    ) {
        // Resize and initialize output matrix
        const unsigned int n_cells = std::get<1>(MeshUtilities<2>::CalculateMeshData(rBoxDivisions));
        if (rOutput.size() != n_cells) {
            // This shall be done BEFORE creating the buffer for GPU
            // rOutput.resize(n_cells);
        }

        // Get the cell gradient operator
        // Eigen::Array<double,4,2> cell_gradient_operator;
        double cell_gradient_operator_data[8];
        IncompressibleNavierStokesQ1P0StructuredElement::QuadVectorDataView cell_gradient_operator(cell_gradient_operator_data, 4, 2);
        IncompressibleNavierStokesQ1P0StructuredElement::GetCellGradientOperator(rCellSize[0], rCellSize[1], cell_gradient_operator);

        // Apply the gradient operator onto a vector
        std::array<int,4> cell_node_ids;
        for (unsigned int i = 0; i < rBoxDivisions[1]; ++i) {
            for (unsigned int j = 0; j < rBoxDivisions[0]; ++j) {
                const unsigned int cell_id = CellUtilities::GetCellGlobalId(i, j, rBoxDivisions);
                double& r_val = rOutput[cell_id];
                r_val = 0.0;
                if (rActiveCells[cell_id]) {
                    CellUtilities::GetCellNodesGlobalIds(i, j, rBoxDivisions, cell_node_ids);
                    for (unsigned int  d = 0; d < 2; ++d) {
                        for (unsigned int i_node = 0; i_node < 4; ++i_node) {
                            r_val += cell_gradient_operator(i_node, d) * rX(cell_node_ids[i_node], d);
                        }
                    }
                }
            }
        }
    }

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
        MatrixViewType aux(mAuxData, mNnodes, TDim);

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
    }

    template<class src_accessor_t, class dst_accessor_t>
    SYCL_EXTERNAL void ApplyGPU(
        const src_accessor_t& rInput,
              dst_accessor_t& rOutput)
    {
        MatrixViewType aux(mAuxData, mNnodes, TDim);

        // Apply gradient operator to input vector
        LocalApplyGradientOperator(mrBoxDivisions, mrCellSize, mrActiveCells, rInput, aux);

        // Apply the lumped mass inverse to the auxiliary vector
        for (unsigned int i = 0; i < mrLumpedMassVectorInv.extent(0); ++i) {
            for (unsigned int j = 0; j < mrLumpedMassVectorInv.extent(1); ++j) {
                aux(i, j) *= mrLumpedMassVectorInv(i, j);
            }
        }

        // Apply the divergence operator to the auxiliary vector and store the result in the output array
        LocalApplyDivergenceOperator(mrBoxDivisions, mrCellSize, mrActiveCells, aux, rOutput);
    }

    void ApplyAccessor(auto& rInput, auto& rOutput)
    {
        MatrixViewType aux(mAuxData, mNnodes, TDim);

        // Apply gradient operator to input vector
        LocalApplyGradientOperator(mrBoxDivisions, mrCellSize, mrActiveCells, rInput, aux);

        // Apply the lumped mass inverse to the auxiliary vector
        for (unsigned int i = 0; i < mrLumpedMassVectorInv.extent(0); ++i) {
            for (unsigned int j = 0; j < mrLumpedMassVectorInv.extent(1); ++j) {
                aux(i, j) *= mrLumpedMassVectorInv(i, j);
            }
        }

        // Apply the divergence operator to the auxiliary vector and store the result in the output array
        LocalApplyDivergenceOperator(mrBoxDivisions, mrCellSize, mrActiveCells, aux, rOutput);
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

    double * mAuxData;

    const std::array<double, TDim>& mrCellSize;

    const std::array<int, TDim>& mrBoxDivisions;

    const std::vector<bool>& mrActiveCells;

    const MatrixViewType& mrLumpedMassVectorInv;

    const int mNnodes;

};

#undef TDim 