#include <fstream>
#include <iostream>

#include "cell_utilities.hpp"
#include "mesh_utilities.hpp"
#include "incompressible_navier_stokes_q1_p0_structured_element.hpp"
#include "operators.hpp"

template<>
void Operators<2>::ApplyGradientOperator(
        const std::array<int, 2>& rBoxDivisions,
        const std::array<double, 2>& rCellSize,
        const std::vector<double>& rX,
        MatrixViewType& rOutput)
{
    // Check output matrix sizes (i.e. mdspan extent)
    const unsigned int n_nodes = std::get<0>(MeshUtilities<2>::CalculateMeshData(rBoxDivisions));
    if (rOutput.extent(0) != n_nodes || rOutput.extent(1) != 2) {
        throw std::logic_error("Wrong size in mdspan extent.");
    }

    // Initialize output matrix
    for (unsigned int i = 0; i < rOutput.extent(0); ++i) {
        rOutput(i, 0) = 0.0;
        rOutput(i, 1) = 0.0;
    }

    // Get the cell gradient operator
    Eigen::Array<double,4,2> cell_gradient_operator;
    IncompressibleNavierStokesQ1P0StructuredElement::GetCellGradientOperator(rCellSize[0], rCellSize[1], cell_gradient_operator);

    // Apply the gradient operator onto a vector
    std::array<int,4> cell_node_ids;
    for (unsigned int i = 0; i < rBoxDivisions[1]; ++i) {
        for (unsigned int j = 0; j < rBoxDivisions[0]; ++j) {
            const double x = rX[CellUtilities::GetCellGlobalId(i, j, rBoxDivisions)];
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

template<>
void Operators<3>::ApplyGradientOperator(
        const std::array<int, 3>& rBoxDivisions,
        const std::array<double, 3>& rCellSize,
        const std::vector<double>& rX,
        MatrixViewType& rOutput)
{
    // Check output matrix sizes (i.e. mdspan extent)
    const unsigned int n_nodes = std::get<0>(MeshUtilities<3>::CalculateMeshData(rBoxDivisions));
    if (rOutput.extent(0) != n_nodes || rOutput.extent(1) != 3) {
        throw std::logic_error("Wrong size in mdspan extent.");
    }

    // Initialize output matrix
    for (unsigned int i = 0; i < rOutput.extent(0); ++i) {
        rOutput(i, 0) = 0.0;
        rOutput(i, 1) = 0.0;
        rOutput(i, 2) = 0.0;
    }

    // Get the cell gradient operator
    Eigen::Array<double,8,3> cell_gradient_operator;
    IncompressibleNavierStokesQ1P0StructuredElement::GetCellGradientOperator(rCellSize[0], rCellSize[1], rCellSize[2], cell_gradient_operator);

    // Apply the gradient operator onto a vector
    std::array<int,8> cell_node_ids;
    for (unsigned int i = 0; i < rBoxDivisions[1]; ++i) {
        for (unsigned int j = 0; j < rBoxDivisions[0]; ++j) {
            for (unsigned int k = 0; k < rBoxDivisions[2]; ++k) {
                const double x = rX[CellUtilities::GetCellGlobalId(i, j, k, rBoxDivisions)];
                CellUtilities::GetCellNodesGlobalIds(i, j, k, rBoxDivisions, cell_node_ids);
                unsigned int i_node = 0;
                for (unsigned int id_node : cell_node_ids) {
                    rOutput(id_node, 0) += cell_gradient_operator(i_node, 0) * x;
                    rOutput(id_node, 1) += cell_gradient_operator(i_node, 1) * x;
                    rOutput(id_node, 2) += cell_gradient_operator(i_node, 2) * x;
                    i_node++;
                }
            }
        }
    }
}

template<>
void Operators<2>::ApplyGradientOperator(
        const std::array<int, 2>& rBoxDivisions,
        const std::array<double, 2>& rCellSize,
        const std::vector<bool>& rActiveCells,
        const std::vector<double>& rX,
        MatrixViewType& rOutput)
{
    // Check output matrix sizes (i.e. mdspan extent)
    const unsigned int n_nodes = std::get<0>(MeshUtilities<2>::CalculateMeshData(rBoxDivisions));
    if (rOutput.extent(0) != n_nodes || rOutput.extent(1) != 2) {
        throw std::logic_error("Wrong size in mdspan extent.");
    }

    // Initialize output matrix
    for (unsigned int i = 0; i < rOutput.extent(0); ++i) {
        rOutput(i, 0) = 0.0;
        rOutput(i, 1) = 0.0;
    }

    // Get the cell gradient operator
    Eigen::Array<double,4,2> cell_gradient_operator;
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

template<>
void Operators<3>::ApplyGradientOperator(
        const std::array<int, 3>& rBoxDivisions,
        const std::array<double, 3>& rCellSize,
        const std::vector<bool>& rActiveCells,
        const std::vector<double>& rX,
        MatrixViewType& rOutput)
{
    // Check output matrix sizes (i.e. mdspan extent)
    const unsigned int n_nodes = std::get<0>(MeshUtilities<3>::CalculateMeshData(rBoxDivisions));
    if (rOutput.extent(0) != n_nodes || rOutput.extent(1) != 3) {
        throw std::logic_error("Wrong size in mdspan extent.");
    }

    // Initialize output matrix
    for (unsigned int i = 0; i < rOutput.extent(0); ++i) {
        rOutput(i, 0) = 0.0;
        rOutput(i, 1) = 0.0;
        rOutput(i, 2) = 0.0;
    }

    // Get the cell gradient operator
    Eigen::Array<double,8,3> cell_gradient_operator;
    IncompressibleNavierStokesQ1P0StructuredElement::GetCellGradientOperator(rCellSize[0], rCellSize[1], rCellSize[2], cell_gradient_operator);

    // Apply the gradient operator onto a vector
    std::array<int,8> cell_node_ids;
    for (unsigned int i = 0; i < rBoxDivisions[1]; ++i) {
        for (unsigned int j = 0; j < rBoxDivisions[0]; ++j) {
            for (unsigned int k = 0; k < rBoxDivisions[2]; ++k) {
                const unsigned int cell_id = CellUtilities::GetCellGlobalId(i, j, k, rBoxDivisions);
                if (rActiveCells[cell_id]) {
                    const double x = rX[cell_id];
                    CellUtilities::GetCellNodesGlobalIds(i, j, k, rBoxDivisions, cell_node_ids);
                    unsigned int i_node = 0;
                    for (unsigned int id_node : cell_node_ids) {
                        rOutput(id_node, 0) += cell_gradient_operator(i_node, 0) * x;
                        rOutput(id_node, 1) += cell_gradient_operator(i_node, 1) * x;
                        rOutput(id_node, 2) += cell_gradient_operator(i_node, 2) * x;
                        i_node++;
                    }
                }
            }
        }
    }
}

template<>
void Operators<2>::ApplyDivergenceOperator(
    const std::array<int, 2>& rBoxDivisions,
    const std::array<double, 2>& rCellSize,
    const MatrixViewType& rX,
    std::vector<double>& rOutput)
{
    // Resize and initialize output matrix
    const unsigned int n_cells = std::get<1>(MeshUtilities<2>::CalculateMeshData(rBoxDivisions));
    if (rOutput.size() != n_cells) {
        rOutput.resize(n_cells);
    }

    // Get the cell gradient operator
    Eigen::Array<double,4,2> cell_gradient_operator;
    IncompressibleNavierStokesQ1P0StructuredElement::GetCellGradientOperator(rCellSize[0], rCellSize[1], cell_gradient_operator);

    // Apply the gradient operator onto a vector
    std::array<int,4> cell_node_ids;
    for (unsigned int i = 0; i < rBoxDivisions[1]; ++i) {
        for (unsigned int j = 0; j < rBoxDivisions[0]; ++j) {
            const unsigned int cell_id = CellUtilities::GetCellGlobalId(i, j, rBoxDivisions);
            double& r_val = rOutput[cell_id];
            r_val = 0.0;
            CellUtilities::GetCellNodesGlobalIds(i, j, rBoxDivisions, cell_node_ids);
            for (unsigned int  d = 0; d < 2; ++d) {
                for (unsigned int i_node = 0; i_node < 4; ++i_node) {
                    r_val += cell_gradient_operator(i_node, d) * rX(cell_node_ids[i_node], d);
                }
            }
        }
    }
}

template<>
void Operators<3>::ApplyDivergenceOperator(
        const std::array<int, 3>& rBoxDivisions,
        const std::array<double, 3>& rCellSize,
        const MatrixViewType& rX,
        std::vector<double>& rOutput)
{
    // Resize and initialize output matrix
    const unsigned int n_cells = std::get<1>(MeshUtilities<3>::CalculateMeshData(rBoxDivisions));
    if (rOutput.size() != n_cells) {
        rOutput.resize(n_cells);
    }

    // Get the cell gradient operator
    Eigen::Array<double,8,3> cell_gradient_operator;
    IncompressibleNavierStokesQ1P0StructuredElement::GetCellGradientOperator(rCellSize[0], rCellSize[1], rCellSize[2], cell_gradient_operator);

    // Apply the gradient operator onto a vector
    std::array<int,8> cell_node_ids;
    for (unsigned int i = 0; i < rBoxDivisions[1]; ++i) {
        for (unsigned int j = 0; j < rBoxDivisions[0]; ++j) {
            for (unsigned int k = 0; k < rBoxDivisions[2]; ++k) {
                const unsigned int cell_id = CellUtilities::GetCellGlobalId(i, j, k, rBoxDivisions);
                double& r_val = rOutput[cell_id];
                r_val = 0.0;
                CellUtilities::GetCellNodesGlobalIds(i, j, k, rBoxDivisions, cell_node_ids);
                for (unsigned int  d = 0; d < 3; ++d) {
                    for (unsigned int i_node = 0; i_node < 8; ++i_node) {
                        r_val += cell_gradient_operator(i_node, d) * rX(cell_node_ids[i_node], d);
                    }
                }
            }
        }
    }
}

template<>
void Operators<2>::ApplyDivergenceOperator(
    const std::array<int, 2>& rBoxDivisions,
    const std::array<double, 2>& rCellSize,
    const std::vector<bool>& rActiveCells,
    const MatrixViewType& rX,
    std::vector<double>& rOutput)
{
    // Resize and initialize output matrix
    const unsigned int n_cells = std::get<1>(MeshUtilities<2>::CalculateMeshData(rBoxDivisions));
    if (rOutput.size() != n_cells) {
        rOutput.resize(n_cells);
    }

    // Get the cell gradient operator
    Eigen::Array<double,4,2> cell_gradient_operator;
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

template<>
void Operators<3>::ApplyDivergenceOperator(
        const std::array<int, 3>& rBoxDivisions,
        const std::array<double, 3>& rCellSize,
        const std::vector<bool>& rActiveCells,
        const MatrixViewType& rX,
        std::vector<double>& rOutput)
{
    // Resize and initialize output matrix
    const unsigned int n_cells = std::get<1>(MeshUtilities<3>::CalculateMeshData(rBoxDivisions));
    if (rOutput.size() != n_cells) {
        rOutput.resize(n_cells);
    }

    // Get the cell gradient operator
    Eigen::Array<double,8,3> cell_gradient_operator;
    IncompressibleNavierStokesQ1P0StructuredElement::GetCellGradientOperator(rCellSize[0], rCellSize[1], rCellSize[2], cell_gradient_operator);

    // Apply the gradient operator onto a vector
    std::array<int,8> cell_node_ids;
    for (unsigned int i = 0; i < rBoxDivisions[1]; ++i) {
        for (unsigned int j = 0; j < rBoxDivisions[0]; ++j) {
            for (unsigned int k = 0; k < rBoxDivisions[2]; ++k) {
                const unsigned int cell_id = CellUtilities::GetCellGlobalId(i, j, k, rBoxDivisions);
                double& r_val = rOutput[cell_id];
                r_val = 0.0;
                if (rActiveCells[cell_id]) {
                    CellUtilities::GetCellNodesGlobalIds(i, j, k, rBoxDivisions, cell_node_ids);
                    for (unsigned int  d = 0; d < 3; ++d) {
                        for (unsigned int i_node = 0; i_node < 8; ++i_node) {
                            r_val += cell_gradient_operator(i_node, d) * rX(cell_node_ids[i_node], d);
                        }
                    }
                }
            }
        }
    }
}

template<int TDim>
void Operators<TDim>::ApplyPressureOperator(
    const std::array<int, TDim>& rBoxDivisions,
    const std::array<double, TDim>& rCellSize,
    const MatrixViewType& rLumpedMassVectorInv,
    const std::vector<double>& rX,
    std::vector<double>& rOutput)
{
    const unsigned int n_nodes = std::get<0>(MeshUtilities<TDim>::CalculateMeshData(rBoxDivisions));
    double aux_data[n_nodes * TDim];
    MatrixViewType aux(aux_data, n_nodes, TDim);
    ApplyGradientOperator(rBoxDivisions, rCellSize, rX, aux);
    for (unsigned int i = 0; i < aux.extent(0); ++i) {
        for (unsigned int j = 0; j < aux.extent(1); ++j) {
            aux(i, j) *= rLumpedMassVectorInv(i, j);
        }
    }
    ApplyDivergenceOperator(rBoxDivisions, rCellSize, aux, rOutput);
}

template<int TDim>
void Operators<TDim>::OutputPressureOperator(
    const std::array<int, TDim>& rBoxDivisions,
    const std::array<double, TDim>& rCellSize,
    const MatrixViewType& rLumpedMassVectorInv,
    const std::string Filename,
    const std::string OutputPath)
{
    // Allocate auxiliary data
    const unsigned int n_cells = std::get<1>(MeshUtilities<TDim>::CalculateMeshData(rBoxDivisions));
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
        ApplyPressureOperator(rBoxDivisions, rCellSize, rLumpedMassVectorInv, aux_vect, r_out_col);
    }

    // Get the non-zero entries of the pressure operator
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

template<int TDim>
void Operators<TDim>::ApplyPressureOperator(
    const std::array<int, TDim>& rBoxDivisions,
    const std::array<double, TDim>& rCellSize,
    const std::vector<bool>& rActiveCells,
    const MatrixViewType& rLumpedMassVectorInv,
    const std::vector<double>& rX,
    std::vector<double>& rOutput)
{
    const unsigned int n_nodes = std::get<0>(MeshUtilities<TDim>::CalculateMeshData(rBoxDivisions));
    double aux_data[n_nodes * TDim];
    MatrixViewType aux(aux_data, n_nodes, TDim);
    ApplyGradientOperator(rBoxDivisions, rCellSize, rActiveCells, rX, aux);
    for (unsigned int i = 0; i < aux.extent(0); ++i) {
        for (unsigned int j = 0; j < aux.extent(1); ++j) {
            aux(i, j) *= rLumpedMassVectorInv(i, j);
        }
    }
    ApplyDivergenceOperator(rBoxDivisions, rCellSize, rActiveCells, aux, rOutput);
}

template<int TDim>
void Operators<TDim>::OutputPressureOperator(
    const std::array<int, TDim>& rBoxDivisions,
    const std::array<double, TDim>& rCellSize,
    const std::vector<bool>& rActiveCells,
    const MatrixViewType& rLumpedMassVectorInv,
    const std::string Filename,
    const std::string OutputPath)
{
// Allocate auxiliary data
    const unsigned int n_cells = std::get<1>(MeshUtilities<TDim>::CalculateMeshData(rBoxDivisions));
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
        ApplyPressureOperator(rBoxDivisions, rCellSize, rActiveCells, rLumpedMassVectorInv, aux_vect, r_out_col);
    }

    // Get the non-zero entries of the pressure operator
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

template class Operators<2>;
template class Operators<3>;
