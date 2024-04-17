#include <iostream>

#include "cell_utilities.hpp"
#include "mesh_utilities.hpp"
#include "incompressible_navier_stokes_q1_p0_structured_element.hpp"
#include "operators.hpp"

template<>
void Operators<2>::ApplyGradientOperator(
        const std::array<int, 2>& rBoxDivisions,
        const std::array<double, 2>& rCellSize,
        const std::vector<bool>& rActiveCells,
        const std::vector<double>& rX,
        Eigen::Matrix<double, Eigen::Dynamic, 2>& rOutput)
{
    // Resize and initialize output matrix
    const unsigned int n_nodes = std::get<0>(MeshUtilities<2>::CalculateMeshData(rBoxDivisions));
    if (rOutput.rows() != n_nodes) {
        rOutput.resize(n_nodes, 2);
    }
    rOutput.setZero();

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
                    auto r_node_row = rOutput.row(id_node);
                    r_node_row(0) += cell_gradient_operator(i_node, 0) * x;
                    r_node_row(1) += cell_gradient_operator(i_node, 1) * x;
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
        Eigen::Matrix<double, Eigen::Dynamic, 3>& rOutput)
{
    // Resize and initialize output matrix
    const unsigned int n_nodes = std::get<0>(MeshUtilities<3>::CalculateMeshData(rBoxDivisions));
    if (rOutput.rows() != n_nodes) {
        rOutput.resize(n_nodes, 3);
    }
    rOutput.setZero();

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
                        auto r_node_row = rOutput.row(id_node);
                        r_node_row(0) += cell_gradient_operator(i_node, 0) * x;
                        r_node_row(1) += cell_gradient_operator(i_node, 1) * x;
                        r_node_row(2) += cell_gradient_operator(i_node, 2) * x;
                        i_node++;
                    }
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
    const std::vector<bool>& rActiveCells,
    const Eigen::Array<double, Eigen::Dynamic, 2>& rX,
    std::vector<double>& rOutput)
{
    // Resize and initialize output matrix
    const unsigned int n_cells = std::get<1>(MeshUtilities<2>::CalculateMeshData(rBoxDivisions));
    if (rOutput.size() != n_cells) {
        rOutput.resize(n_cells);
    }
    std::fill(rOutput.begin(), rOutput.end(), 0.0); //TODO: Avoid the std::fill to make this parallel

    // Get the cell gradient operator
    Eigen::Array<double,4,2> cell_gradient_operator;
    IncompressibleNavierStokesQ1P0StructuredElement::GetCellGradientOperator(rCellSize[0], rCellSize[1], cell_gradient_operator);

    // Apply the gradient operator onto a vector
    std::array<int,4> cell_node_ids;
    for (unsigned int i = 0; i < rBoxDivisions[1]; ++i) {
        for (unsigned int j = 0; j < rBoxDivisions[0]; ++j) {
            const unsigned int cell_id = CellUtilities::GetCellGlobalId(i, j, rBoxDivisions);
            if (rActiveCells[cell_id]) {
                CellUtilities::GetCellNodesGlobalIds(i, j, rBoxDivisions, cell_node_ids);
                auto cell_x = rX(cell_node_ids, Eigen::all);
                for (unsigned int d = 0; d < 2; ++d) {
                    rOutput[cell_id] += (cell_gradient_operator.col(d) * cell_x.col(d)).sum();
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
        const Eigen::Array<double, Eigen::Dynamic, 3>& rX,
        std::vector<double>& rOutput)
{
    // Resize and initialize output matrix
    const unsigned int n_cells = std::get<1>(MeshUtilities<3>::CalculateMeshData(rBoxDivisions));
    if (rOutput.size() != n_cells) {
        rOutput.resize(n_cells);
    }
    std::fill(rOutput.begin(), rOutput.end(), 0.0); //TODO: Avoid the std::fill to make this parallel

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
                    CellUtilities::GetCellNodesGlobalIds(i, j, k, rBoxDivisions, cell_node_ids);
                    auto cell_x = rX(cell_node_ids, Eigen::all);
                    for (unsigned int d = 0; d < 3; ++d) {
                        rOutput[cell_id] += (cell_gradient_operator.col(d) * cell_x.col(d)).sum();
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
    const std::vector<bool>& rActiveCells,
    const Eigen::Array<double, Eigen::Dynamic, TDim>& rLumpedMassVectorInv,
    const std::vector<double>& rX,
    std::vector<double>& rOutput)
{
    Eigen::Matrix<double, Eigen::Dynamic, TDim> aux;
    ApplyGradientOperator(rBoxDivisions, rCellSize, rActiveCells, rX, aux);
    aux.array() *= rLumpedMassVectorInv;
    ApplyDivergenceOperator(rBoxDivisions, rCellSize, rActiveCells, aux, rOutput);
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

template class Operators<2>;
template class Operators<3>;
