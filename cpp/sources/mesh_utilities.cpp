#include <iostream>

#include "cell_utilities.hpp"
#include "mesh_utilities.hpp"

template<int TDim>
std::pair<unsigned int, unsigned int> MeshUtilities<TDim>::CalculateMeshData(const std::array<int, TDim>& rBoxDivisions)
{
    if constexpr (TDim == 2) {
        const unsigned int num_cells = rBoxDivisions[0] * rBoxDivisions[1];
        const unsigned int num_nodes = (rBoxDivisions[0] + 1) * (rBoxDivisions[1] + 1);
        return std::make_pair(num_nodes, num_cells);
    } else {
        const unsigned int num_cells = rBoxDivisions[0] * rBoxDivisions[1] * rBoxDivisions[2];
        const unsigned int num_nodes = (rBoxDivisions[0] + 1) * (rBoxDivisions[1] + 1) * (rBoxDivisions[2] + 1);
        return std::make_pair(num_nodes, num_cells);
    }
}

template<int TDim>
std::array<double, TDim> MeshUtilities<TDim>::CalculateCellSize(
    const std::array<double, TDim>& rBoxSize,
    const std::array<int, TDim>& rBoxDivisions)
{
    std::array<double, TDim> cell_size;
    cell_size[0] = rBoxSize[0] / rBoxDivisions[0];
    cell_size[1] = rBoxSize[1] / rBoxDivisions[1];
    if constexpr (TDim == 3) {
        cell_size[1] = rBoxSize[2] / rBoxDivisions[2];
    }
    return cell_size;
}

template<int TDim>
void MeshUtilities<TDim>::CalculateNodalCoordinates(
    const std::array<double, TDim>& rBoxSize,
    const std::array<int, TDim>& rBoxDivisions,
    Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic>& rNodalCoords)
{
    // Allocate coordinates vector
    const unsigned int num_nodes = std::get<0>(CalculateMeshData(rBoxDivisions));
    rNodalCoords.resize(num_nodes, TDim);

    // Get mesh parameters
    const unsigned int n_x = rBoxDivisions[0] + 1;
    const unsigned int n_y = rBoxDivisions[1] + 1;
    const unsigned int n_z = TDim == 3 ? rBoxDivisions[2] + 1 : 0;
    const double l_x = rBoxSize[0];
    const double l_y = rBoxSize[1];
    const double l_z = TDim == 3 ? rBoxSize[2] : 0.0;

    // Generate nodal coordinates
    std::vector<double> x_coords(n_x);
    std::vector<double> y_coords(n_y);
    std::vector<double> z_coords(n_z);

	const double step_x = l_x / rBoxDivisions[0];
    for (unsigned int i_x = 0; i_x < n_x; ++i_x) {
        x_coords[i_x] = i_x * step_x;
    }

	const double step_y = l_y / rBoxDivisions[1];
    for (unsigned int i_y = 0; i_y < n_y; ++i_y) {
        y_coords[i_y] = i_y * step_y;
    }

    if constexpr (TDim == 3) {
        const double step_z = l_z / rBoxDivisions[2];
        for (unsigned int i_z = 0; i_z < n_z; ++i_z) {
            z_coords[i_z] = i_z * step_z;
        }
    }

    // Fill the coordinates array
    if constexpr (TDim == 2) {
        unsigned int aux = 0;
        for (const double y : y_coords) {
            for (const double x : x_coords) {
                rNodalCoords(aux, 0) = x;
                rNodalCoords(aux, 1) = y;
                ++aux;
            }
        }
    } else {
        unsigned int aux = 0;
        for (const double z : z_coords) {
            for (const double y : y_coords) {
                for (const double x : x_coords) {
                    rNodalCoords(aux, 0) = x;
                    rNodalCoords(aux, 1) = y;
                    rNodalCoords(aux, 2) = z;
                    ++aux;
                }
            }
        }
    }
}

template<>
void MeshUtilities<2>::CalculateLumpedMassVector(
    const double MassFactor,
    const std::array<int, 2>& rBoxDivisions,
    const Eigen::Array<bool, Eigen::Dynamic, 1>& rActiveCells,
    Eigen::Array<double, Eigen::Dynamic, 2>& rLumpedMassVector)
{
    const unsigned int num_nodes = std::get<0>(CalculateMeshData(rBoxDivisions));
    if (rLumpedMassVector.rows() != num_nodes) {
        rLumpedMassVector.resize(num_nodes, 2);
    }
    rLumpedMassVector.setZero();
    std::array<int, 4> cell_node_ids;
    for (unsigned int i = 0; i < rBoxDivisions[0]; ++i) {
        for (unsigned int j = 0; j < rBoxDivisions[1]; ++j) {
            if (rActiveCells(CellUtilities::GetCellGlobalId(i, j, rBoxDivisions))) {
                CellUtilities::GetCellNodesGlobalIds(i, j, rBoxDivisions, cell_node_ids);
                rLumpedMassVector(cell_node_ids, Eigen::all).array() += MassFactor;
            }
        }
    }
}

template<>
void MeshUtilities<3>::CalculateLumpedMassVector(
    const double MassFactor,
    const std::array<int, 3>& rBoxDivisions,
    const Eigen::Array<bool, Eigen::Dynamic, 1>& rActiveCells,
    Eigen::Array<double, Eigen::Dynamic, 3>& rLumpedMassVector)
{
    const unsigned int num_nodes = std::get<0>(CalculateMeshData(rBoxDivisions));
    if (rLumpedMassVector.rows() != num_nodes) {
        rLumpedMassVector.resize(num_nodes, 3);
    }
    rLumpedMassVector.setZero();
    std::array<int, 8> cell_node_ids;
    for (unsigned int i = 0; i < rBoxDivisions[0]; ++i) {
        for (unsigned int j = 0; j < rBoxDivisions[1]; ++j) {
            for (unsigned int k = 0; k < rBoxDivisions[2]; ++k) {
                if (rActiveCells(CellUtilities::GetCellGlobalId(i, j, k, rBoxDivisions))) {
                    CellUtilities::GetCellNodesGlobalIds(i, j, k, rBoxDivisions, cell_node_ids);
                    rLumpedMassVector(cell_node_ids, Eigen::all).array() += MassFactor;
                }
            }
        }
    }
}

template <>
unsigned int MeshUtilities<2>::FindFirstFreeCellId(
    const std::array<int, 2> &rBoxDivisions,
    const Eigen::Array<bool, Eigen::Dynamic, 2>& rFixity)
{
    std::array<int, 4> cell_node_ids;
    for (unsigned int i = 0; i < rBoxDivisions[0]; ++i) {
        for (unsigned int j = 0; j < rBoxDivisions[1]; ++j) {
            CellUtilities::GetCellNodesGlobalIds(i, j, rBoxDivisions, cell_node_ids);
            const auto cell_fixity = rFixity(cell_node_ids, Eigen::all);
            if (!cell_fixity.any()) {
                return CellUtilities::GetCellGlobalId(i, j, rBoxDivisions);
            }
        }
    }
    throw std::runtime_error("There is no cell with all the DOFs free.");
    return 0;
}

template <>
unsigned int MeshUtilities<3>::FindFirstFreeCellId(
    const std::array<int, 3> &rBoxDivisions,
    const Eigen::Array<bool, Eigen::Dynamic, 3>& rFixity)
{
    std::array<int, 8> cell_node_ids;
    for (unsigned int i = 0; i < rBoxDivisions[0]; ++i) {
        for (unsigned int j = 0; j < rBoxDivisions[1]; ++j) {
            for (unsigned int k = 0; k < rBoxDivisions[2]; ++k) {
                CellUtilities::GetCellNodesGlobalIds(i, j, k, rBoxDivisions, cell_node_ids);
                const auto cell_fixity = rFixity(cell_node_ids, Eigen::all);
                if (!cell_fixity.any()) {
                    return CellUtilities::GetCellGlobalId(i, j, k, rBoxDivisions);
                }
            }
        }
    }
    throw std::runtime_error("There is no cell with all the DOFs free.");
    return 0;
}

template class MeshUtilities<2>;
template class MeshUtilities<3>;
