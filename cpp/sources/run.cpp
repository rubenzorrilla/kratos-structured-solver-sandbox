#include <array>
#include <iostream>
#include <Eigen/Dense>

#include "cell_utilities.hpp"
#include "incompressible_navier_stokes_q1_p0_structured_element.hpp"
#include "mesh_utilities.hpp"
#include "runge_kutta_utilities.hpp"
#include "time_utilities.hpp"

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
    // const std::array<int, dim> box_divisions({150, 30});
    const std::array<int, dim> box_divisions({4, 1});

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
    Eigen::ArrayXXd p = Eigen::VectorXd::Zero(num_cells);
    Eigen::ArrayXXd f = Eigen::MatrixXd::Zero(num_nodes, dim);
    Eigen::ArrayXXd v = Eigen::MatrixXd::Zero(num_nodes, dim);
    Eigen::ArrayXXd v_n = Eigen::MatrixXd::Zero(num_nodes, dim);
    Eigen::ArrayXXd acc = Eigen::MatrixXd::Zero(num_nodes, dim);

    // Set velocity fixity vector and BCs
    // Note that these overwrite the initial conditions above
    Eigen::Array<bool, Eigen::Dynamic, dim> fixity(num_nodes, dim);
    fixity.setZero();

    const double tol = 1.0e-6;
    for (unsigned int i_node = 0; i_node < num_nodes; ++i_node) {
        const auto& r_coords = nodal_coords.row(i_node);
        // Inlet
        if (r_coords[0] < tol) {
            fixity(i_node, 0) = true; // x-velocity
            fixity(i_node, 1) = true; // y-velocity
            const double y_coord = r_coords(1);
            // v(i_node, 0) = 6.0*y_coord*(1.0-y_coord);
            // v_n(i_node, 0) = 6.0*y_coord*(1.0-y_coord);
            v(i_node, 0) = 1.0;
            v_n(i_node, 0) = 1.0;
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
        // distance(i_node) = dist < cyl_rad ? - dist : dist;
        distance(i_node) = 1.0;
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
    Eigen::Array<bool, Eigen::Dynamic, 1> active_cells(num_cells);
    if constexpr (dim == 2) {
        std::array<int,4> cell_node_ids;
        for (unsigned int i = 0; i < box_divisions[0]; ++i) {
            for (unsigned int j = 0; j < box_divisions[1]; ++j) {
                CellUtilities::GetCellNodesGlobalIds(i, j, box_divisions, cell_node_ids);
                const auto cell_fixity = fixity(cell_node_ids, Eigen::all);
                active_cells(CellUtilities::GetCellGlobalId(i, j, box_divisions)) = !cell_fixity.all();
            }
        }
    } else {
        throw std::logic_error("3D case not implemented yet");
    }

    // Set forcing term
    f.setZero();

    // Set final free/fixed DOFs arrays
    // TODO: Check if there is a more efficient way to do this
    std::vector<unsigned int> free_dofs_rows;
    std::vector<unsigned int> free_dofs_cols;
    std::vector<unsigned int> fixed_dofs_rows;
    std::vector<unsigned int> fixed_dofs_cols;
    for (unsigned int i_row = 0; i_row < fixity.rows(); ++i_row) {
        for (unsigned int j_col = 0; j_col < fixity.cols(); ++j_col) {
            if (fixity(i_row, j_col)) {
                fixed_dofs_rows.push_back(i_row);
                fixed_dofs_cols.push_back(j_col);
            } else {
                free_dofs_rows.push_back(i_row);
                free_dofs_cols.push_back(j_col);
            }
        }
    }

    // Calculate lumped mass vector
    const double cell_domain_size = CellUtilities::GetCellDomainSize(cell_size);
    const double mass_factor = rho * cell_domain_size / (dim == 2 ? 4.0 : 8.0);
    Eigen::Array<double, Eigen::Dynamic, dim> lumped_mass_vector(num_nodes, dim);
    MeshUtilities<dim>::CalculateLumpedMassVector(mass_factor, box_divisions, active_cells, lumped_mass_vector);

    // Calculate inverse of the lumped mass vector
    Eigen::Array<double, Eigen::Dynamic, dim> lumped_mass_vector_inv(num_nodes, dim);
    for (unsigned int i_node = 0; i_node < num_nodes; ++i_node) {
        if (lumped_mass_vector(i_node, 0) > 0.0) {
            lumped_mass_vector_inv(i_node, Eigen::all) = 1.0 / lumped_mass_vector(i_node, 0);
        } else {
            lumped_mass_vector_inv(i_node, Eigen::all) = 0.0;
        }
    }
    Eigen::Array<double, Eigen::Dynamic, dim> lumped_mass_vector_inv_bcs = lumped_mass_vector_inv;
    lumped_mass_vector_inv_bcs(fixed_dofs_rows, fixed_dofs_cols) = 0.0;

    // Set Runge-Kutta arrays
    constexpr int rk_order = 4;
    Eigen::Array<double, rk_order, 1> rk_B;
    Eigen::Array<double, rk_order, 1> rk_C;
    Eigen::Array<double, rk_order, rk_order> rk_A;
    RungeKuttaUtilities<rk_order>::SetNodesVector(rk_C);
    RungeKuttaUtilities<rk_order>::SetWeightsVector(rk_B);
    RungeKuttaUtilities<rk_order>::SetRungeKuttaMatrix(rk_A);

    // Allocate auxiliary arrays
    constexpr int rk_num_steps = rk_order;
    Eigen::Array<double, Eigen::Dynamic, dim> rk_v(num_nodes, dim);
    std::array<Eigen::Array<double, Eigen::Dynamic, dim>, rk_num_steps> rk_res;
    for (auto& r_arr : rk_res) {
        r_arr.resize(num_nodes, dim);
    }

    // Time loop
    unsigned int tot_p_iters = 0;
    unsigned int current_step = 1;
    double current_time = init_time;
    while (current_time < end_time) {
        // Compute time increment with CFL and Fourier conditions
        // Note that we use the current step velocity to be updated as it equals the previous one at this point
        const double dt = TimeUtilities<dim>::CalculateDeltaTime(rho, mu, cell_size, v, 0.2, 0.2);
        std::cout << "### Step " << current_step << " - time " << current_time << " - dt " << dt << " ###" << std::endl;

        // Update the surrogate boundary Dirichlet value from the previous time step velocity gradient
        //TODO:

        // Calculate intermediate residuals
        for (unsigned int rk_step = 0; rk_step < rk_num_steps; ++rk_step) {
            // Initialize current intermediate step residual
            auto& r_rk_step_res = rk_res[rk_step];
            r_rk_step_res.setZero();

            // Calculate current step velocity for residual calculation
            const double rk_theta = rk_C[rk_step];
            const double rk_step_time = current_time + rk_theta * dt;
            rk_v.setZero();
            for (unsigned int i_step = 0; i_step < rk_step; ++i_step) {
                const double a_ij = rk_A(rk_step, i_step);
                rk_v += a_ij * rk_res[i_step];
            }
            rk_v *= dt * lumped_mass_vector_inv;
            rk_v += v_n;
            rk_v(fixed_dofs_rows, fixed_dofs_cols) = rk_theta * v(fixed_dofs_rows, fixed_dofs_cols) + (1.0 - rk_theta) * v_n(fixed_dofs_rows, fixed_dofs_cols); // Set BC value in fixed DOFs

            // Calculate current step residual
            if constexpr (dim == 2) {
                std::array<int,4> cell_node_ids;
                Eigen::Array<double, 4, 2> cell_v;
                Eigen::Array<double, 4, 2> cell_f;
                Eigen::Array<double, 4, 2> cell_acc;
                Eigen::Array<double, 8, 1> cell_res;
                for (unsigned int i = 0; i < box_divisions[0]; ++i) {
                    for (unsigned int j = 0; j < box_divisions[1]; ++j) {
                        const unsigned int i_cell = CellUtilities::GetCellGlobalId(i, j, box_divisions);
                        if (active_cells(i_cell)) {
                            // Get current cell data
                            CellUtilities::GetCellNodesGlobalIds(i, j, box_divisions, cell_node_ids);
                            const double cell_p = p(i_cell);
                            cell_v = rk_v(cell_node_ids, Eigen::all);
                            cell_f = f(cell_node_ids, Eigen::all);
                            cell_acc = acc(cell_node_ids, Eigen::all);

                            // Calculate current cell residual
                            IncompressibleNavierStokesQ1P0StructuredElement::CalculateRightHandSide(cell_size[0], cell_size[1], mu, rho, cell_v, cell_p, cell_f, cell_acc, cell_res);
                            if (i_cell == 1) {
                                std::cout << "i_cell: " << i_cell << std::endl;
                                std::cout << cell_v << std::endl;
                                std::cout << cell_p << std::endl;
                                std::cout << cell_res << std::endl;
                            }

                            // Assemble current cell residual
                            unsigned int aux_i = 0;
                            for (const int id_node : cell_node_ids) {
                                for (unsigned int d = 0; d < 2; ++d) {
                                    rk_res[rk_step](id_node, d) += cell_res(2 * aux_i + d);
                                }
                                aux_i++;
                            }
                        }
                    }
                }
            } else {
                throw std::logic_error("3D case not implemented yet");
            }
        }

        // Solve Runge-Kutta step
        v(free_dofs_rows, free_dofs_cols) = 0.0;
        for (unsigned int rk_step = 0; rk_step < rk_num_steps; ++rk_step) {
            v(free_dofs_rows, free_dofs_cols) += rk_B[rk_step] * rk_res[rk_step](free_dofs_rows, free_dofs_cols);
        }
        v(free_dofs_rows, free_dofs_cols) *= dt * lumped_mass_vector_inv(free_dofs_rows, free_dofs_cols);
        v(free_dofs_rows, free_dofs_cols) += v_n(free_dofs_rows, free_dofs_cols);
        std::cout << "Velocity prediction solved." << std::endl;

        // Update variables for next time step
        acc = (v - v_n) / dt;
        v_n = v;
        ++current_step;
        current_time += dt;

        if (current_step == 2) {
            break;
        }
    }

    std::cout << "v: \n" <<  v << std::endl;
    std::cout << "p: \n" << p << std::endl;

    return 0;
}