#include <iostream>

#include "cell_utilities.hpp"
#include "incompressible_navier_stokes_q1_p0_structured_element.hpp"
#include "mesh_utilities.hpp"
#include "sbm_utilities.hpp"

template <>
void SbmUtilities<2>::UpdateSurrogateBoundaryDirichletValues(
    const double MassFactor,
    const std::array<int, 2> &rBoxDivisions,
    const std::array<double, 2> &rCellSize,
    const Eigen::Array<bool, Eigen::Dynamic, 1> &rSurrogateCells,
    const Eigen::Array<bool, Eigen::Dynamic, 1> &rSurrogateNodes,
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
    for (unsigned int i = 0; i < rBoxDivisions[0]; ++i) {
        for (unsigned int j = 0; j < rBoxDivisions[1]; ++j) {
            // Check if current cell is attached to the surrogate boundary
            if (rSurrogateCells(CellUtilities::GetCellGlobalId(i, j, rBoxDivisions))) {
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
                    if (rSurrogateNodes(node_id)) {
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
    const Eigen::Array<bool, Eigen::Dynamic, 1> &rSurrogateCells,
    const Eigen::Array<bool, Eigen::Dynamic, 1> &rSurrogateNodes,
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
    for (unsigned int i = 0; i < rBoxDivisions[0]; ++i) {
        for (unsigned int j = 0; j < rBoxDivisions[1]; ++j) {
            for (unsigned int k = 0; k < rBoxDivisions[2]; ++k) {
                // Check if current cell is attached to the surrogate boundary
                if (rSurrogateCells(CellUtilities::GetCellGlobalId(i, j, k, rBoxDivisions))) {
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
                        if (rSurrogateNodes(node_id)) {
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
