#include <array>
#include <iostream>
#include <Eigen/Dense>

#include "cell_utilities.hpp"
#include "mesh_utilities.hpp"

int main()
{
    // Problem data
    static constexpr int dim = 2;
    const double end_time = 1.0e1;
    const double init_time = 0.0;

    // Material data
    const double mu = 2.0e-3;
    const double rho = 1.0e0;

    // Input mesh data
    const std::array<double, dim> box_size({5.0, 1.0});
    const std::array<int, dim> box_divisions({150, 30});

    // Compute mesh data
    auto mesh_data = MeshUtilities<dim>::CalculateMeshData(box_divisions);
    auto cell_size = MeshUtilities<dim>::CalculateCellSize(box_size, box_divisions);
    const int num_nodes = std::get<0>(mesh_data);
    const int num_cells = std::get<1>(mesh_data);

    std::cout << "### MESH DATA ###" << std::endl;
    std::cout << "num_nodes: " << num_nodes << std::endl;
    std::cout << "num_cells: " << num_cells << std::endl;
    if constexpr (dim == 2) {
        std::cout << "cell_size: [" << cell_size[0] << "," << cell_size[1] << "]" << std::endl;
    } else {
        std::cout << "cell_size: [" << cell_size[0] << "," << cell_size[1] << "," << cell_size[2] <<  "]" << std::endl;
    }

    // Create mesh nodes (only used for fixity and visualization)
    Eigen::ArrayXXd nodal_coords;
    MeshUtilities<dim>::CalculateNodalCoordinates(box_size, box_divisions, nodal_coords);

    // Create mesh dataset
    Eigen::ArrayXXd p = Eigen::VectorXd::Zero(num_nodes);
    Eigen::ArrayXXd f = Eigen::MatrixXd::Zero(num_nodes, dim);
    Eigen::ArrayXXd v = Eigen::MatrixXd::Zero(num_nodes, dim);
    Eigen::ArrayXXd v_n = Eigen::MatrixXd::Zero(num_nodes, dim);
    Eigen::ArrayXXd acc = Eigen::MatrixXd::Zero(num_nodes, dim);

    // Set velocity fixity vector and BCs
    // Note that these overwrite the initial conditions above
    Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic> fixity(num_nodes, dim);
    fixity.setZero();

    const double tol = 1.0e-6;
    for (unsigned int i_node = 0; i_node < num_nodes; ++i_node) {
        const auto& r_coords = nodal_coords.row(i_node);
        // Inlet
        if (r_coords[0] < tol) {
            fixity(i_node, 0) = true; // x-velocity
            fixity(i_node, 1) = true; // y-velocity
            const double y_coord = r_coords(1);
            v(i_node, 0) = 6.0*y_coord*(1.0-y_coord);
            v_n(i_node, 0) = 6.0*y_coord*(1.0-y_coord);
        }

        // Top and bottom wals
        if ((r_coords[1] < tol) || (r_coords[1] > (1.0-tol))) {
            fixity(i_node, 1) = true; // y-velocity
        }
    }

    // Calculate the distance values
    const double cyl_rad = 0.1;
    Eigen::Array<double, 1, dim> cyl_orig;
    cyl_orig(0) = 1.25;
    cyl_orig(1) = 0.5;

    Eigen::Array<double, Eigen::Dynamic, 1> distance(num_nodes);
    for (unsigned int i_node = 0; i_node < num_nodes; ++i_node) {
        const auto& r_coords = nodal_coords.row(i_node);
        const double dist = (r_coords - cyl_orig).matrix().norm();
        distance(i_node) = dist < cyl_rad ? - dist : dist;
    }

    // Find the surrogate boundary
    Eigen::Array<bool, Eigen::Dynamic, 1> surrogate_nodes(num_nodes);
    surrogate_nodes.setZero();
    if constexpr (dim == 2) {
        std::array<int,4> cell_node_ids;
        for (unsigned int i = 0; i < box_divisions[0]; ++i) {
            for (unsigned int j = 0; j < box_divisions[1]; ++j) {
                CellUtilities::GetCellNodesGlobalIds(i, j, box_divisions, cell_node_ids);
                std::vector<unsigned int> pos_nodes;
                std::vector<unsigned int> neg_nodes;
                const auto cell_distance = distance(cell_node_ids, 0);
                for (unsigned int i_node = 0; i_node < 4; ++i_node) {
                    if (cell_distance[i_node] < 0.0) {
                        neg_nodes.push_back(cell_node_ids[i_node]);
                    } else {
                        pos_nodes.push_back(cell_node_ids[i_node]);
                    }
                }
                if (pos_nodes.size() != 0 && neg_nodes.size() != 0) {
                    surrogate_nodes(pos_nodes, 0) = true;
                }
            }
        }
    } else {
        throw std::logic_error("3D case not implemented yet");
    }

    Eigen::Array<bool, Eigen::Dynamic, 1> surrogate_cells(num_cells);
    surrogate_cells.setZero();
    if constexpr (dim == 2) {
        std::array<int,4> cell_node_ids;
        for (unsigned int i = 0; i < box_divisions[0]; ++i) {
            for (unsigned int j = 0; j < box_divisions[1]; ++j) {
                CellUtilities::GetCellNodesGlobalIds(i, j, box_divisions, cell_node_ids);
                const auto& r_cell_node_dist = distance(cell_node_ids);
                if (std::none_of(r_cell_node_dist.cbegin(), r_cell_node_dist.cend(), [](const double x){return x < 0.0;})) {
                    for (int node_id : cell_node_ids) {
                        if (surrogate_nodes(node_id)) {
                            surrogate_cells(CellUtilities::GetCellGlobalId(i, j, box_divisions)) = true;
                            break;
                        }
                    }
                }
            }
        }
    } else {
        throw std::logic_error("3D case not implemented yet");
    }

    // Calculate the distance vectors in the surrogate boundary nodes
    Eigen::Vector<double, dim> aux_dir;
    Eigen::Array<double, Eigen::Dynamic, dim> distance_vects(num_nodes, dim);
    for (unsigned int i_node = 0; i_node < num_nodes; ++i_node) {
        if (surrogate_nodes(i_node)) {
            const auto& r_coords = nodal_coords.row(i_node);
            aux_dir = cyl_orig - r_coords;
            aux_dir /= aux_dir.norm();
            distance_vects.row(i_node) = distance[i_node] * aux_dir;
        }
    }

    // Apply fixity in the interior nodes
    for (unsigned int i_node = 0; i_node < num_nodes; ++i_node) {
        if (distance[i_node] < 0.0) {
            fixity.row(i_node).setConstant(true);
        }
    }

    // Apply fixity in the surrogate boundary nodes
    for (unsigned int i_node = 0; i_node < num_nodes; ++i_node) {
        if (surrogate_nodes(i_node)) {
            fixity.row(i_node).setConstant(true);
        }
    }

    // Deactivate the interior and intersected cells (SBM-like resolution)
    if constexpr (dim == 2) {
        std::array<int,4> cell_node_ids;
        Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic> active_cells(box_divisions[0], box_divisions[1]);
        for (unsigned int i = 0; i < box_divisions[0]; ++i) {
            for (unsigned int j = 0; j < box_divisions[1]; ++j) {
                CellUtilities::GetCellNodesGlobalIds(i, j, box_divisions, cell_node_ids);
                const auto cell_fixity = fixity(cell_node_ids, Eigen::all);
                bool is_fixed = true;
                for (unsigned int i_fix = 0; i_fix < cell_fixity.rows(); ++i_fix) {
                    for (unsigned int j_fix = 0; j_fix < cell_fixity.cols(); ++j_fix) {
                        if (!cell_fixity(i_fix, j_fix)) {
                            is_fixed = false;
                            break;
                        }
                    }
                }
                active_cells(i, j) = is_fixed;
            }
        }
    } else {
        throw std::logic_error("3D case not implemented yet");
    }

    // Set forcing term
    f.setZero();



    return 0;
}