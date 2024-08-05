#include <fstream>
#include <iomanip>
#include <iostream>

#include "cell_utilities.hpp"
#include "mdspan_utilities.hpp"
#include "mesh_utilities.hpp"

// template<int TDim>
// std::pair<unsigned int, unsigned int> MeshUtilities<TDim>::CalculateMeshData(const std::array<int, TDim>& rBoxDivisions)
// {
//     if constexpr (TDim == 2) {
//         const unsigned int num_cells = rBoxDivisions[0] * rBoxDivisions[1];
//         const unsigned int num_nodes = (rBoxDivisions[0] + 1) * (rBoxDivisions[1] + 1);
//         return std::make_pair(num_nodes, num_cells);
//     } else {
//         const unsigned int num_cells = rBoxDivisions[0] * rBoxDivisions[1] * rBoxDivisions[2];
//         const unsigned int num_nodes = (rBoxDivisions[0] + 1) * (rBoxDivisions[1] + 1) * (rBoxDivisions[2] + 1);
//         return std::make_pair(num_nodes, num_cells);
//     }
// }

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
    MatrixViewType& rNodalCoords)
{
    // Check coordinates view extent
    const unsigned int num_nodes = std::get<0>(CalculateMeshData(rBoxDivisions));
    if (rNodalCoords.extent(0) != num_nodes || rNodalCoords.extent(1) != TDim) {
        // throw std::logic_error("Wrong size in mdspan extent.");
    }

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

// template<int TDim>
// void MeshUtilities<TDim>::CalculateNodalCoordinatesGPU(
//     const std::array<double, TDim>& rBoxSize,
//     const std::array<int, TDim>& rBoxDivisions,
//     MatrixViewType& rNodalCoords)
// {
//     // Check coordinates view extent
//     const unsigned int num_nodes = std::get<0>(CalculateMeshData(rBoxDivisions));
//     if (rNodalCoords.extent(0) != num_nodes || rNodalCoords.extent(1) != TDim) {
//         // throw std::logic_error("Wrong size in mdspan extent.");
//     }

//     // Get mesh parameters
//     const unsigned int n_x = rBoxDivisions[0] + 1;
//     const unsigned int n_y = rBoxDivisions[1] + 1;
//     const unsigned int n_z = TDim == 3 ? rBoxDivisions[2] + 1 : 0;
//     const double l_x = rBoxSize[0];
//     const double l_y = rBoxSize[1];
//     const double l_z = TDim == 3 ? rBoxSize[2] : 0.0;

//     // Generate nodal coordinates
//     std::vector<double> x_coords(n_x);
//     std::vector<double> y_coords(n_y);
//     std::vector<double> z_coords(n_z);

// 	const double step_x = l_x / rBoxDivisions[0];
//     for (unsigned int i_x = 0; i_x < n_x; ++i_x) {
//         x_coords[i_x] = i_x * step_x;
//     }

// 	const double step_y = l_y / rBoxDivisions[1];
//     for (unsigned int i_y = 0; i_y < n_y; ++i_y) {
//         y_coords[i_y] = i_y * step_y;
//     }

//     if constexpr (TDim == 3) {
//         const double step_z = l_z / rBoxDivisions[2];
//         for (unsigned int i_z = 0; i_z < n_z; ++i_z) {
//             z_coords[i_z] = i_z * step_z;
//         }
//     }

//     // Fill the coordinates array
//     if constexpr (TDim == 2) {
//         unsigned int aux = 0;
//         for (const double y : y_coords) {
//             for (const double x : x_coords) {
//                 rNodalCoords(aux, 0) = x;
//                 rNodalCoords(aux, 1) = y;
//                 ++aux;
//             }
//         }
//     } else {
//         unsigned int aux = 0;
//         for (const double z : z_coords) {
//             for (const double y : y_coords) {
//                 for (const double x : x_coords) {
//                     rNodalCoords(aux, 0) = x;
//                     rNodalCoords(aux, 1) = y;
//                     rNodalCoords(aux, 2) = z;
//                     ++aux;
//                 }
//             }
//         }
//     }
// }

template<>
void MeshUtilities<2>::CalculateLumpedMassVector(
    const double MassFactor,
    const std::array<int, 2>& rBoxDivisions,
    MatrixViewType& rLumpedMassVector)
{
    const unsigned int num_nodes = std::get<0>(CalculateMeshData(rBoxDivisions));
    if (rLumpedMassVector.extent(0) != num_nodes || rLumpedMassVector.extent(1) != 2) {
        // throw std::logic_error("Wrong size in mdspan extent.");
    }
    MdspanUtilities::SetZero(rLumpedMassVector);
    std::array<int, 4> cell_node_ids;
    for (unsigned int i = 0; i < rBoxDivisions[1]; ++i) {
        for (unsigned int j = 0; j < rBoxDivisions[0]; ++j) {
            CellUtilities::GetCellNodesGlobalIds(i, j, rBoxDivisions, cell_node_ids);
            for (unsigned int node_id : cell_node_ids) {
                for (unsigned int d = 0; d < 2; ++d) {
                    rLumpedMassVector(node_id, d) += MassFactor;
                }
            }
        }
    }
}

template<>
void MeshUtilities<3>::CalculateLumpedMassVector(
    const double MassFactor,
    const std::array<int, 3>& rBoxDivisions,
    MatrixViewType& rLumpedMassVector)
{
    const unsigned int num_nodes = std::get<0>(CalculateMeshData(rBoxDivisions));
    if (rLumpedMassVector.extent(0) != num_nodes || rLumpedMassVector.extent(1) != 3) {
        // throw std::logic_error("Wrong size in mdspan extent.");
    }
    MdspanUtilities::SetZero(rLumpedMassVector);
    std::array<int, 8> cell_node_ids;
    for (unsigned int i = 0; i < rBoxDivisions[1]; ++i) {
        for (unsigned int j = 0; j < rBoxDivisions[0]; ++j) {
            for (unsigned int k = 0; k < rBoxDivisions[2]; ++k) {
                CellUtilities::GetCellNodesGlobalIds(i, j, k, rBoxDivisions, cell_node_ids);
                for (unsigned int node_id : cell_node_ids) {
                    for (unsigned int d = 0; d < 3; ++d) {
                        rLumpedMassVector(node_id, d) += MassFactor;
                    }
                }
            }
        }
    }
}

template<>
void MeshUtilities<2>::CalculateLumpedMassVector(
    const double MassFactor,
    const std::array<int, 2>& rBoxDivisions,
    const std::vector<bool>& rActiveCells,
    MatrixViewType& rLumpedMassVector)
{
    const unsigned int num_nodes = std::get<0>(CalculateMeshData(rBoxDivisions));
    if (rLumpedMassVector.extent(0) != num_nodes || rLumpedMassVector.extent(1) != 2) {
        // throw std::logic_error("Wrong size in mdspan extent.");
    }
    MdspanUtilities::SetZero(rLumpedMassVector);
    std::array<int, 4> cell_node_ids;
    for (unsigned int i = 0; i < rBoxDivisions[1]; ++i) {
        for (unsigned int j = 0; j < rBoxDivisions[0]; ++j) {
            if (rActiveCells[CellUtilities::GetCellGlobalId(i, j, rBoxDivisions)]) {
                CellUtilities::GetCellNodesGlobalIds(i, j, rBoxDivisions, cell_node_ids);
                for (unsigned int node_id : cell_node_ids) {
                    for (unsigned int d = 0; d < 2; ++d) {
                        rLumpedMassVector(node_id, d) += MassFactor;
                    }
                }
            }
        }
    }
}

template<>
void MeshUtilities<3>::CalculateLumpedMassVector(
    const double MassFactor,
    const std::array<int, 3>& rBoxDivisions,
    const std::vector<bool>& rActiveCells,
    MatrixViewType& rLumpedMassVector)
{
    const unsigned int num_nodes = std::get<0>(CalculateMeshData(rBoxDivisions));
    if (rLumpedMassVector.extent(0) != num_nodes || rLumpedMassVector.extent(1) != 3) {
        // throw std::logic_error("Wrong size in mdspan extent.");
    }
    MdspanUtilities::SetZero(rLumpedMassVector);
    std::array<int, 8> cell_node_ids;
    for (unsigned int i = 0; i < rBoxDivisions[1]; ++i) {
        for (unsigned int j = 0; j < rBoxDivisions[0]; ++j) {
            for (unsigned int k = 0; k < rBoxDivisions[2]; ++k) {
                if (rActiveCells[CellUtilities::GetCellGlobalId(i, j, k, rBoxDivisions)]) {
                    CellUtilities::GetCellNodesGlobalIds(i, j, k, rBoxDivisions, cell_node_ids);
                    for (unsigned int node_id : cell_node_ids) {
                        for (unsigned int d = 0; d < 3; ++d) {
                            rLumpedMassVector(node_id, d) += MassFactor;
                        }
                    }
                }
            }
        }
    }
}

template <>
std::tuple<bool, unsigned int> MeshUtilities<2>::FindFirstFreeCellId(
    const std::array<int, 2> &rBoxDivisions,
    const FixityMatrixViewType& rFixity,
    const std::vector<bool>& rActiveCells)
{
    std::array<int, 8> neigh_cells_ids;
    std::array<int, cell_nodes> cell_node_ids;
    for (unsigned int i = 0; i < rBoxDivisions[1]; ++i) {
        for (unsigned int j = 0; j < rBoxDivisions[0]; ++j) {
            // Get current cell nodal ids
            CellUtilities::GetCellNodesGlobalIds(i, j, rBoxDivisions, cell_node_ids);

            // Get the number of free DOFs in current cell
            unsigned int n_free_dofs = 0;
            for (unsigned int node_id : cell_node_ids) {
                for (unsigned int d = 0; d < 2; ++d) {
                    if (!rFixity(node_id, d)) {
                        n_free_dofs++;
                    }
                }
            }

            // Check if the number of free DOFs matches the total number of DOFs in the cell
            if (n_free_dofs == cell_dofs) {
                // Get the neighbouring cells global ids
                CellUtilities::GetNeighbourCellsGlobalIds(i, j, rBoxDivisions, neigh_cells_ids);

                // If all DOFs in the cell are free, check that it has complete stencil
                // That is to check that it is completely surrounded by active neighbour cells
                bool valid_neighs = true;
                for (int neigh_id : neigh_cells_ids) {
                    if (neigh_id == -1) { // Note that if there is no neighbour the id is set to -1
                        valid_neighs = false;
                        break;
                    } else {
                        if (!rActiveCells[neigh_id]) { // Check if the neighbour cell is active
                            valid_neighs = false;
                            break;
                        }
                    }
                }

                // If all the neighbouring cells are available and active return the id of current cell
                if (valid_neighs) {
                    return std::make_tuple(true, CellUtilities::GetCellGlobalId(i, j, rBoxDivisions));
                }
            }
        }
    }

    return std::make_tuple(false, 0);
}

template <>
std::tuple<bool, unsigned int> MeshUtilities<3>::FindFirstFreeCellId(
    const std::array<int, 3> &rBoxDivisions,
    const FixityMatrixViewType& rFixity,
    const std::vector<bool>& rActiveCells)
{
    std::array<int, 24> neigh_cells_ids;
    std::array<int, cell_nodes> cell_node_ids;
    for (unsigned int i = 0; i < rBoxDivisions[1]; ++i) {
        for (unsigned int j = 0; j < rBoxDivisions[0]; ++j) {
            for (unsigned int k = 0; k < rBoxDivisions[2]; ++k) {
                // Get current cell nodal ids
                CellUtilities::GetCellNodesGlobalIds(i, j, k, rBoxDivisions, cell_node_ids);

                // Get the number of free DOFs in current cell
                unsigned int n_free_dofs = 0;
                for (unsigned int node_id : cell_node_ids) {
                    for (unsigned int d = 0; d < 3; ++d) {
                        if (!rFixity(node_id, d)) {
                            n_free_dofs++;
                        }
                    }
                }

                // Check if the number of free DOFs matches the total number of DOFs in the cell
                if (n_free_dofs == cell_dofs) {
                    // Get the neighbouring cells global ids
                    CellUtilities::GetNeighbourCellsGlobalIds(i, j, k, rBoxDivisions, neigh_cells_ids);

                    // If all DOFs in the cell are free, check that it has complete stencil
                    // That is to check that it is completely surrounded by active neighbour cells
                    bool valid_neighs = true;
                    for (int neigh_id : neigh_cells_ids) {
                        if (neigh_id == -1) { // Note that if there is no neighbour the id is set to -1
                            valid_neighs = false;
                            break;
                        } else {
                            if (!rActiveCells[neigh_id]) { // Check if the neighbour cell is active
                                valid_neighs = false;
                                break;
                            }
                        }
                    }

                    // If all the neighbouring cells are available and active return the id of current cell
                    if (valid_neighs) {
                        return std::make_tuple(true, CellUtilities::GetCellGlobalId(i, j, k, rBoxDivisions));
                    }
                }
            }
        }
    }

    return std::make_tuple(false, 0);
}

template<int TDim>
void MeshUtilities<TDim>::OutputVector(
    const std::vector<double>& rVector,
    const std::string Filename,
    const std::string OutputPath)
{
    const double aux_tol = 1.0e-14;
    std::vector<std::tuple<unsigned int, double>> non_zero_entries;
    for (unsigned int i = 0; i < rVector.size(); ++i) {
        if (std::abs(rVector[i]) > aux_tol) {
            non_zero_entries.push_back(std::make_tuple(i, rVector[i]));
        }
    }

    std::ofstream out_file(OutputPath + Filename + ".mm");
    if (out_file.is_open()) {
        out_file <<  std::scientific << std::setprecision(12);
        out_file << "%%MatrixMarket matrix coordinate real general" << std::endl;
        out_file << rVector.size() << "  " << 1 << "  " << non_zero_entries.size() << std::endl;
        for (auto& r_non_zero_entry : non_zero_entries) {
            const unsigned int row = std::get<0>(r_non_zero_entry);
            const double val = std::get<1>(r_non_zero_entry);
            out_file << row + 1 << "  " << 1 << "  " << val << std::endl;
        }
        out_file.close();
    }
}

template class MeshUtilities<2>;
template class MeshUtilities<3>;
