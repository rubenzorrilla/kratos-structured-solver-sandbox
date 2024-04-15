#include <iostream>

#include "cell_utilities.hpp"
#include "incompressible_navier_stokes_q1_p0_structured_element.hpp"
#include "mesh_utilities.hpp"
#include "sbm_utilities.hpp"

template <>
void SbmUtilities<2>::FindSurrogateBoundaryNodes(
    const std::array<int, 2> &rBoxDivisions,
    const std::vector<double>& rDistance,
    std::vector<bool>& rSurrogateNodes)
{
    const unsigned int num_nodes = std::get<0>(MeshUtilities<2>::CalculateMeshData(rBoxDivisions));
    if (rSurrogateNodes.size() != num_nodes) {
        rSurrogateNodes.resize(num_nodes);
    }
    std::fill(rSurrogateNodes.begin(), rSurrogateNodes.end(), false); //TODO: Avoid std::fill to make it parallel

    std::array<int,4> cell_node_ids;
    for (unsigned int i = 0; i < rBoxDivisions[1]; ++i) {
        for (unsigned int j = 0; j < rBoxDivisions[0]; ++j) {
            CellUtilities::GetCellNodesGlobalIds(i, j, rBoxDivisions, cell_node_ids);
            std::vector<unsigned int> pos_nodes;
            std::vector<unsigned int> neg_nodes;
            for (unsigned int i_node = 0; i_node < 4; ++i_node) {
                const unsigned int node_id = cell_node_ids[i_node];
                if (rDistance[node_id] < 0.0) {
                    neg_nodes.push_back(node_id);
                } else {
                    pos_nodes.push_back(node_id);
                }
            }
            if (pos_nodes.size() != 0 && neg_nodes.size() != 0) {
                for (unsigned int i = 0; i < pos_nodes.size(); ++i) {
                    rSurrogateNodes[pos_nodes[i]] = true;
                }
            }
        }
    }
}

template <>
void SbmUtilities<3>::FindSurrogateBoundaryNodes(
    const std::array<int, 3> &rBoxDivisions,
    const std::vector<double>& rDistance,
    std::vector<bool>& rSurrogateNodes)
{
    const unsigned int num_nodes = std::get<0>(MeshUtilities<3>::CalculateMeshData(rBoxDivisions));
    if (rSurrogateNodes.size() != num_nodes) {
        rSurrogateNodes.resize(num_nodes);
    }
    std::fill(rSurrogateNodes.begin(), rSurrogateNodes.end(), false); //TODO: Avoid std::fill to make it parallel

    std::array<int,8> cell_node_ids;
    for (unsigned int i = 0; i < rBoxDivisions[1]; ++i) {
        for (unsigned int j = 0; j < rBoxDivisions[0]; ++j) {
            for (unsigned int k = 0; k < rBoxDivisions[2]; ++k) {
                CellUtilities::GetCellNodesGlobalIds(i, j, k, rBoxDivisions, cell_node_ids);
                std::vector<unsigned int> pos_nodes;
                std::vector<unsigned int> neg_nodes;
                for (unsigned int i_node = 0; i_node < 8; ++i_node) {
                    const unsigned int node_id = cell_node_ids[i_node];
                    if (rDistance[node_id] < 0.0) {
                        neg_nodes.push_back(cell_node_ids[i_node]);
                    } else {
                        pos_nodes.push_back(cell_node_ids[i_node]);
                    }
                }
                if (pos_nodes.size() != 0 && neg_nodes.size() != 0) {
                    for (unsigned int i = 0; i < pos_nodes.size(); ++i) {
                        rSurrogateNodes[pos_nodes[i]] = true;
                    }
                }
            }
        }
    }
}

template <>
void SbmUtilities<2>::FindSurrogateBoundaryCells(
    const std::array<int, 2> &rBoxDivisions,
    const std::vector<double>& rDistance,
    const std::vector<bool>& rSurrogateNodes,
    std::vector<bool>& rSurrogateCells)
{
    const unsigned int num_cells = std::get<1>(MeshUtilities<2>::CalculateMeshData(rBoxDivisions));
    if (rSurrogateCells.size() != num_cells) {
        rSurrogateCells.resize(num_cells);
    }
    std::fill(rSurrogateCells.begin(), rSurrogateCells.end(), false); //TODO: Avoid std::fill to make it parallel

    std::array<int,4> cell_node_ids;
    std::array<int,4> cell_node_dist;
    for (unsigned int i = 0; i < rBoxDivisions[1]; ++i) {
        for (unsigned int j = 0; j < rBoxDivisions[0]; ++j) {
            CellUtilities::GetCellNodesGlobalIds(i, j, rBoxDivisions, cell_node_ids);
            for (unsigned int i = 0; i < 4; ++i) {
                cell_node_dist[i] = rDistance[cell_node_ids[i]];
            }
            if (std::none_of(cell_node_dist.cbegin(), cell_node_dist.cend(), [](const double x){return x < 0.0;})) {
                for (int node_id : cell_node_ids) {
                    if (rSurrogateNodes[node_id]) {
                        rSurrogateCells[CellUtilities::GetCellGlobalId(i, j, rBoxDivisions)] = true;
                        break;
                    }
                }
            }
        }
    }
}

template <>
void SbmUtilities<3>::FindSurrogateBoundaryCells(
    const std::array<int, 3> &rBoxDivisions,
    const std::vector<double>& rDistance,
    const std::vector<bool>& rSurrogateNodes,
    std::vector<bool>& rSurrogateCells)
{
    const unsigned int num_cells = std::get<1>(MeshUtilities<3>::CalculateMeshData(rBoxDivisions));
    if (rSurrogateCells.size() != num_cells) {
        rSurrogateCells.resize(num_cells);
    }
    std::fill(rSurrogateCells.begin(), rSurrogateCells.end(), false); //TODO: Avoid std::fill to make it parallel

    std::array<int,8> cell_node_ids;
    std::array<int,8> cell_node_dist;
    for (unsigned int i = 0; i < rBoxDivisions[1]; ++i) {
        for (unsigned int j = 0; j < rBoxDivisions[0]; ++j) {
            for (unsigned int k = 0; k < rBoxDivisions[2]; ++k) {
                CellUtilities::GetCellNodesGlobalIds(i, j, k, rBoxDivisions, cell_node_ids);
                for (unsigned int i = 0; i < 4; ++i) {
                    cell_node_dist[i] = rDistance[cell_node_ids[i]];
                }
                if (std::none_of(cell_node_dist.cbegin(), cell_node_dist.cend(), [](const double x){return x < 0.0;})) {
                    for (int node_id : cell_node_ids) {
                        if (rSurrogateNodes[node_id]) {
                            rSurrogateCells[CellUtilities::GetCellGlobalId(i, j, k, rBoxDivisions)] = true;
                            break;
                        }
                    }
                }
            }
        }
    }
}

template <>
void SbmUtilities<2>::UpdateSurrogateBoundaryDirichletValues(
    const double MassFactor,
    const std::array<int, 2> &rBoxDivisions,
    const std::array<double, 2> &rCellSize,
    const std::vector<bool> &rSurrogateCells,
    const std::vector<bool> &rSurrogateNodes,
    const Eigen::Array<double, Eigen::Dynamic, 2> &rLumpedMassVector,
    const Eigen::Array<double, Eigen::Dynamic, 2> &rDistanceVects,
    const Eigen::Array<double, Eigen::Dynamic, 2> &rVelocity,
    Eigen::Array<double, Eigen::Dynamic, 2> &rSurrogateVelocity)
{
    // Resize and initialize surrogate boundary velocity array
    const int num_nodes = std::get<0>(MeshUtilities<2>::CalculateMeshData(rBoxDivisions));
    if (rSurrogateVelocity.rows() != num_nodes) {
        rSurrogateVelocity.resize(num_nodes, 2);
    }
    rSurrogateVelocity.setZero();

    // Get cell gradient operator
    Eigen::Array<double, 4, 2> cell_gradient_operator;
    IncompressibleNavierStokesQ1P0StructuredElement::GetCellGradientOperator(rCellSize[0], rCellSize[1], cell_gradient_operator);

    // Loop cells
    Eigen::Vector2d dir_bc;
    Eigen::Matrix2d weighted_grad_v;
    std::array<int, 4> cell_node_ids;
    Eigen::Matrix<double, 4, 2> cell_v;
    for (unsigned int i = 0; i < rBoxDivisions[1]; ++i) {
        for (unsigned int j = 0; j < rBoxDivisions[0]; ++j) {
            // Check if current cell is attached to the surrogate boundary
            if (rSurrogateCells[CellUtilities::GetCellGlobalId(i, j, rBoxDivisions)]) {
                // Get current surrogate cell nodes
                CellUtilities::GetCellNodesGlobalIds(i, j, rBoxDivisions, cell_node_ids);

                // Get current cell velocities and calculate the weighted gradient
                unsigned int aux_i = 0;
                for (int node_id : cell_node_ids) {
                    cell_v(aux_i, Eigen::all) = rVelocity.row(node_id);
                    aux_i++;
                }
                weighted_grad_v.noalias() = MassFactor * (cell_gradient_operator.matrix().transpose() * cell_v);

                // Calculate the Dirichlet velocity value in the surrogate boundary nodes
                for (int node_id : cell_node_ids) {
                    if (rSurrogateNodes[node_id]) {
                        dir_bc.noalias() = weighted_grad_v * (rDistanceVects.row(node_id).transpose()).matrix();
                        rSurrogateVelocity.row(node_id) -= (dir_bc.transpose().array() / rLumpedMassVector.row(node_id));
                    }
                }
            }
        }
    }
}

template <>
void SbmUtilities<3>::UpdateSurrogateBoundaryDirichletValues(
    const double MassFactor,
    const std::array<int, 3> &rBoxDivisions,
    const std::array<double, 3> &rCellSize,
    const std::vector<bool> &rSurrogateCells,
    const std::vector<bool> &rSurrogateNodes,
    const Eigen::Array<double, Eigen::Dynamic, 3> &rLumpedMassVector,
    const Eigen::Array<double, Eigen::Dynamic, 3> &rDistanceVects,
    const Eigen::Array<double, Eigen::Dynamic, 3> &rVelocity,
    Eigen::Array<double, Eigen::Dynamic, 3> &rSurrogateVelocity)
{
// Resize and initialize surrogate boundary velocity array
    const int num_nodes = std::get<0>(MeshUtilities<3>::CalculateMeshData(rBoxDivisions));
    if (rSurrogateVelocity.rows() != num_nodes) {
        rSurrogateVelocity.resize(num_nodes, 2);
    }
    rSurrogateVelocity.setZero();

    // Get cell gradient operator
    Eigen::Array<double, 8, 3> cell_gradient_operator;
    IncompressibleNavierStokesQ1P0StructuredElement::GetCellGradientOperator(rCellSize[0], rCellSize[1], rCellSize[2], cell_gradient_operator);

    // Loop cells
    Eigen::Vector3d dir_bc;
    Eigen::Matrix3d weighted_grad_v;
    std::array<int, 8> cell_node_ids;
    Eigen::Matrix<double, 8, 3> cell_v;
    for (unsigned int i = 0; i < rBoxDivisions[1]; ++i) {
        for (unsigned int j = 0; j < rBoxDivisions[0]; ++j) {
            for (unsigned int k = 0; k < rBoxDivisions[2]; ++k) {
                // Check if current cell is attached to the surrogate boundary
                if (rSurrogateCells[CellUtilities::GetCellGlobalId(i, j, k, rBoxDivisions)]) {
                    // Get current surrogate cell nodes
                    CellUtilities::GetCellNodesGlobalIds(i, j, k, rBoxDivisions, cell_node_ids);

                    // Get current cell velocities and calculate the weighted gradient
                    unsigned int aux_i = 0;
                    for (int node_id : cell_node_ids) {
                        cell_v(aux_i, Eigen::all) = rVelocity.row(node_id);
                        aux_i++;
                    }
                    weighted_grad_v.noalias() = MassFactor * (cell_gradient_operator.matrix().transpose() * cell_v);

                    // Calculate the Dirichlet velocity value in the surrogate boundary nodes
                    for (int node_id : cell_node_ids) {
                        if (rSurrogateNodes[node_id]) {
                            dir_bc.noalias() = weighted_grad_v * (rDistanceVects.row(node_id).transpose()).matrix();
                            rSurrogateVelocity.row(node_id) -= (dir_bc.transpose().array() / rLumpedMassVector.row(node_id));
                        }
                    }
                }
            }
        }
    }
}

template class SbmUtilities<2>;
template class SbmUtilities<3>;
