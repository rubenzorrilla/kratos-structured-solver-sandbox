#include <array>
#include <string>
#include <sstream>
#include <fstream>
#include <numeric>
#include <iostream>
#include <unistd.h>
#include <sys/stat.h>
#include <filesystem>

#include <Eigen/Dense>
#include <unsupported/Eigen/FFT>
#include "include/experimental/mdspan"

#include "cell_utilities.hpp"
#include "incompressible_navier_stokes_q1_p0_structured_element.hpp"
#include "mesh_utilities.hpp"
#include "operators.hpp"
#include "pressure_conjugate_gradient_solver.hpp"
#include "runge_kutta_utilities.hpp"
#include "sbm_utilities.hpp"
#include "time_utilities.hpp"

int main()
{
    // Problem static data
    static constexpr int dim = 2;
    static constexpr int cell_nodes = dim == 2 ? 4 : 8;

    // Types definition
    //TODO: Find a better way to define these
    using MatrixViewType = Operators<dim>::MatrixViewType;

    // Problem data
    const double end_time = 4.5e1;
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

    // Create results directory
    std::cout << "\n### OUTPUT DATA ###" << std::endl;
    struct stat buffer;
    const std::string results_path = "../cpp_output/";
    if (!(stat(results_path.c_str(), &buffer) == 0 && S_ISDIR(buffer.st_mode))) {
        const int result = mkdir(results_path.c_str(), 0777); // 0777 means full access to everyone
        if (result == 0) {
            std::cout << "Results directory created successfully." << std::endl;
        } else {
            std::cerr << "Failed to create results directory." << std::endl;
        }
    } else {
        std::cout << "Results directory already exists." << std::endl;
    }
    std::cout << "Writing results to: " << results_path << std::endl;

    const unsigned int output_interval = 1;
    std::cout << "Writing interval: " << output_interval << std::endl;

    // Purge previous output
    const bool purge_output = true;
    if (purge_output) {
        // Remove txt and lst files
        for (const auto& entry : std::filesystem::directory_iterator(results_path)) {
            if (entry.path().extension() == ".txt") {
                std::filesystem::remove(entry.path());
            } else if (entry.path().extension() == ".lst") {
                std::filesystem::remove(entry.path());
            }
        }
        // Remove complete directories
        std::filesystem::remove_all(results_path + "gid_output/");
        std::filesystem::remove_all(results_path + "vtu_output/");
    }

    // Create mesh nodes (only used for fixity and visualization)
    std::vector<double> nodal_coords_data(num_nodes * dim);
    MatrixViewType nodal_coords(nodal_coords_data.data(), num_nodes, dim);
    MeshUtilities<dim>::CalculateNodalCoordinates(box_size, box_divisions, nodal_coords);

    // Write coordinates and connectivities for postprocess
    std::ofstream coordinates_file(results_path + "coordinates.txt");
    if (coordinates_file.is_open()) {
        for (unsigned int i = 0; i < num_nodes; ++i) {
            if constexpr (dim == 2) {
                coordinates_file << nodal_coords(i, 0) << " " << nodal_coords(i, 1) << " " << 0.0 << std::endl;
            } else {
                coordinates_file << nodal_coords(i, 0) << " " << nodal_coords(i, 1) << " " << nodal_coords(i, 2) << std::endl;
            }
        }
    }

    std::ofstream connectivities_file(results_path + "connectivities.txt");
    if (connectivities_file.is_open()) {
        if constexpr (dim == 2) {
            std::array<int,4> cell_node_ids;
            for (unsigned int i = 0; i < box_divisions[1]; ++i) {
                for (unsigned int j = 0; j < box_divisions[0]; ++j) {
                    CellUtilities::GetCellNodesGlobalIds(i, j, box_divisions, cell_node_ids);
                    connectivities_file << cell_node_ids[0] << " " << cell_node_ids[1] << " " << cell_node_ids[2] << " " << cell_node_ids[3] << std::endl;
                }
            }
        } else {
            // std::array<int,8> cell_node_ids;
            // for (unsigned int i = 0; i < box_divisions[1]; ++i) {
            //     for (unsigned int j = 0; j < box_divisions[0]; ++j) {
            //         for (unsigned int k = 0; k < box_divisions[2]; ++k) {
            //             CellUtilities::GetCellNodesGlobalIds(i, j, k, box_divisions, cell_node_ids);
            //             connectivities_file << cell_node_ids[0] << " " << cell_node_ids[1] << " " << cell_node_ids[2] << " " << cell_node_ids[3] << cell_node_ids[4] << " " << cell_node_ids[5] << " " << cell_node_ids[6] << " " << cell_node_ids[7] << std::endl;
            //         }
            //     }
            // }
        }
        connectivities_file.close();
    }

    // Create mesh dataset
    std::vector<double> p(num_cells, 0.0);
    std::vector<double> f_data(num_nodes*dim, 0.0);
    std::vector<double> v_data(num_nodes*dim, 0.0);
    std::vector<double> v_n_data(num_nodes*dim, 0.0);
    std::vector<double> acc_data(num_nodes*dim, 0.0);

    // Create mdspan views of the above dataset
    MatrixViewType f(f_data.data(), num_nodes, dim);
    MatrixViewType v(v_data.data(), num_nodes, dim);
    MatrixViewType v_n(v_n_data.data(), num_nodes, dim);
    MatrixViewType acc(acc_data.data(), num_nodes, dim);

    // Set velocity fixity vector and BCs
    // Note that these overwrite the initial conditions above
    Eigen::Array<bool, Eigen::Dynamic, dim> fixity(num_nodes, dim);
    fixity.setZero();

    const double tol = 1.0e-6;
    for (unsigned int i_node = 0; i_node < num_nodes; ++i_node) {
        // Inlet
        if (nodal_coords(i_node, 0) < tol) {
            fixity(i_node, 0) = true; // x-velocity
            fixity(i_node, 1) = true; // y-velocity

            const double y_coord = nodal_coords(i_node, 1);

            v(i_node, 0) = 6.0*y_coord*(1.0-y_coord);
            v_n(i_node, 0) = 6.0*y_coord*(1.0-y_coord);

            // v(i_node, 0) = 1.0;
            // v_n(i_node, 0) = 1.0;

            // if (y_coord > 0.5) {
            //     v(i_node, 0) = y_coord - 0.5;
            //     v_n(i_node, 0) = y_coord - 0.5;
            // }
        }

        // Top wall
        if (nodal_coords(i_node, 1) > (1.0-tol)) {
            fixity(i_node, 1) = true; // y-velocity
        }

        // Bottom wall
        if (nodal_coords(i_node, 1) < tol) {
            fixity(i_node, 1) = true; // y-velocity
        }
    }

    // Calculate the distance values
    const double cyl_rad = 0.1;
    std::array<double, dim> cyl_orig;
    cyl_orig[0] = 1.25;
    cyl_orig[1] = 0.5;

    std::array<double, dim> aux_coords;
    std::vector<double> distance(num_nodes);
    for (unsigned int i_node = 0; i_node < num_nodes; ++i_node) {
        double dist = 0.0;
        for (unsigned int d = 0; d < dim; ++d) {
            dist += std::pow(nodal_coords(i_node, d) - cyl_orig[d], 2);
        }
        dist = std::sqrt(dist);
        distance[i_node] = dist < cyl_rad ? - dist : dist;
        // distance[i_node] = 1.0;
        // distance(i_node) = r_coords[1] - 0.5;
    }

    std::ofstream distance_file(results_path + "distance.txt");
    if (distance_file.is_open()) {
        for (unsigned int i = 0; i < num_nodes; ++i) {
            distance_file << distance[i] << std::endl;
        }
    }

    // Define the surrogate boundary
    std::cout << "\n### SURROGATE BOUNDARY DEFINITION ###" << std::endl;
    std::vector<double> v_surrogate_data(num_nodes * dim);
    SbmUtilities<dim>::MatrixViewType v_surrogate(v_surrogate_data.data(), num_nodes, dim);

    std::vector<bool> surrogate_nodes(num_nodes);
    SbmUtilities<dim>::FindSurrogateBoundaryNodes(box_divisions, distance, surrogate_nodes);
    std::cout << "Surrogate boundary nodes found." << std::endl;

    std::vector<bool> surrogate_cells(num_cells);
    SbmUtilities<dim>::FindSurrogateBoundaryCells(box_divisions, distance, surrogate_nodes, surrogate_cells);
    std::cout << "Surrogate boundary cells found." << std::endl;

    // Calculate the distance vectors in the surrogate boundary nodes
    std::array<double, dim> aux_dir;
    std::vector<double> distance_vects_data(num_nodes * dim, 0.0);
    MatrixViewType distance_vects(distance_vects_data.data(), num_nodes, dim);
    for (unsigned int i_node = 0; i_node < num_nodes; ++i_node) {
        if (surrogate_nodes[i_node]) {
            for (unsigned int d = 0; d < dim; ++d) {
                aux_dir[d] = cyl_orig[d] - nodal_coords(i_node, d);
            }
            const double aux_dir_norm = std::sqrt(std::inner_product(aux_dir.begin(), aux_dir.end(), aux_dir.begin(), 0.0));
            for (unsigned int d = 0; d < dim; ++d) {
                aux_dir[d] /= aux_dir_norm;
                distance_vects(i_node, d) = distance[i_node] * aux_dir[d] / aux_dir_norm;
            }
            // distance_vects(i_node, 0) = 0.0;
            // distance_vects(i_node, 1) = distance[i_node];
        }
    }
    std::cout << "Surrogate boundary nodes distance vectors computed." << std::endl;

    // Apply fixity in the interior nodes
    for (unsigned int i_node = 0; i_node < num_nodes; ++i_node) {
        if (distance[i_node] < 0.0) {
            fixity.row(i_node).setConstant(true);
        }
    }

    // Apply fixity in the surrogate boundary nodes
    for (unsigned int i_node = 0; i_node < num_nodes; ++i_node) {
        if (surrogate_nodes[i_node]) {
            fixity.row(i_node).setConstant(true);
        }
    }

    // Deactivate the interior and intersected cells (SBM-like resolution)
    std::vector<bool> active_cells(num_cells);
    if constexpr (dim == 2) {
        std::array<int,4> cell_node_ids;
        for (unsigned int i = 0; i < box_divisions[1]; ++i) {
            for (unsigned int j = 0; j < box_divisions[0]; ++j) {
                CellUtilities::GetCellNodesGlobalIds(i, j, box_divisions, cell_node_ids);
                const auto cell_fixity = fixity(cell_node_ids, Eigen::all);
                active_cells[CellUtilities::GetCellGlobalId(i, j, box_divisions)] = !cell_fixity.all();
            }
        }
    } else {
        throw std::logic_error("3D case not implemented yet");
    }

    // Set forcing term
    for (unsigned int i = 0; i < num_nodes; ++i) {
        for (unsigned int d = 0; d < dim; ++d) {
            f(i,d) = 0.0;
        }
    }

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
    const unsigned int n_free_dofs = free_dofs_cols.size();
    const unsigned int n_fixed_dofs = fixed_dofs_cols.size();

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
    for (unsigned int i_dof = 0; i_dof < n_fixed_dofs; ++i_dof) {
        lumped_mass_vector_inv_bcs(fixed_dofs_rows[i_dof], fixed_dofs_cols[i_dof]) = 0.0;
    }

    // Create the preconditioner for the pressure CG solver
    // For this we convert the periodic pressure matrix C (with no velocity BCs) to FFT
    // The simplest way is to generate any vector x, compute the image vector y=C*X
    // The transform of C is fft(y)./fft(x), as C is cyclic the transformed C must be diagonal
    // The resulting transformed coefficientes should be real, because the operator is symmetric.
    // Also it should be semidefinite positive (SPD) because the Laplacian operator is SPD
    // The first coefficient is null, because the Laplacian is not PD, just SPD.
    // But we can replace this null coefficient by anything different from 0.
    // At most it would degrade the convergence of the PCG, but we will see that the convergence is OK.
    std::cout << "\n### PRESSURE PRECONDITIONER SET-UP ###" << std::endl;
    std::vector<double> fft_c(num_cells);
    auto free_cell_result = MeshUtilities<dim>::FindFirstFreeCellId(box_divisions, fixity);
    if (std::get<0>(free_cell_result)) {
        const unsigned int free_cell_id = std::get<1>(free_cell_result);
        std::cout << "Free cell id: " << free_cell_id << "." <<std::endl;
        std::vector<double> x(num_cells, 0.0);
        std::vector<double> y(num_cells);
        x[free_cell_id] = 1.0;
        Operators<dim>::ApplyPressureOperator(box_divisions, cell_size, active_cells, lumped_mass_vector_inv, x, y);

        std::vector<std::complex<double>> fft_x(num_cells); // Complex array for FFT(x) output
        std::vector<std::complex<double>> fft_y(num_cells); // Complex array for FFT(y) output
        std::vector<std::complex<double>> x_complex(num_cells); // Complex array for FFT(x) input
        std::vector<std::complex<double>> y_complex(num_cells); // Complex array for FFT(y) input
        for (unsigned int i = 0; i < num_cells; ++i) {
            x_complex[i].real(x[i]); // Set x_complex real data from x
            y_complex[i].real(y[i]); // Set y_complex real data from y
        }

        Eigen::FFT<double> fft;
        fft.fwd(fft_x, x_complex);
        fft.fwd(fft_y, y_complex);
        for (unsigned int i = 0; i < num_cells; ++i) {
            fft_c[i] = (fft_y[i] / fft_x[i]).real(); // Take the real part only (imaginary one is zero)
        }
        fft_c[0] = 1.0; // Remove the first coefficient as this is associated to the solution average
        std::cout << "Pressure preconditioner set." << std::endl;
    } else {
        std::cout << "There is no cell with all the DOFs free. No pressure preconditioner can be set." << std::endl;
    }

    // Set Runge-Kutta arrays
    constexpr int rk_order = 4;
    std::array<double, rk_order> rk_B;
    std::array<double, rk_order> rk_C;
    Eigen::Array<double, rk_order, rk_order> rk_A;
    RungeKuttaUtilities<rk_order>::SetNodesVector(rk_C);
    RungeKuttaUtilities<rk_order>::SetWeightsVector(rk_B);
    RungeKuttaUtilities<rk_order>::SetRungeKuttaMatrix(rk_A);

    // Allocate auxiliary arrays for the velocity problem
    constexpr int rk_num_steps = rk_order;
    Eigen::Array<double, Eigen::Dynamic, dim> rk_v(num_nodes, dim);
    std::array<Eigen::Array<double, Eigen::Dynamic, dim>, rk_num_steps> rk_res;
    for (auto& r_arr : rk_res) {
        r_arr.resize(num_nodes, dim);
    }

    // Allocate auxiliary arrays for the pressure problem
    std::vector<double> delta_p(num_cells);
    std::vector<double> delta_p_rhs(num_cells);

    // Allocate auxiliary arrays for the velocity update
    std::vector<double> delta_p_grad_data(num_nodes * dim, 0.0);
    MatrixViewType delta_p_grad(delta_p_grad_data.data(), num_nodes, dim);

    std::cout << "\n### PRESSURE SOLVER SET-UP ###" << std::endl;
    const double abs_tol = 1.0e-5;
    const double rel_tol = 1.0e-3;
    const double max_iter = 5000;
    PressureOperator<dim> pressure_operator(box_divisions, cell_size, active_cells, lumped_mass_vector_inv_bcs);
    std::cout << "Pressure linear operator created." << std::endl;
    PressureConjugateGradientSolver<dim> cg(abs_tol, rel_tol, max_iter, pressure_operator, fft_c);
    std::cout << "Pressure conjugate gradient solver created." << std::endl;

    // Time loop
    unsigned int tot_p_iters = 0;
    unsigned int current_step = 1;
    double current_time = init_time;
    while (current_time < end_time) {
        // Compute time increment with CFL and Fourier conditions
        // Note that we use the current step velocity to be updated as it equals the previous one at this point
        const double dt = TimeUtilities<dim>::CalculateDeltaTime(rho, mu, cell_size, v, 0.2, 0.2);
        std::cout << "\n### Step " << current_step << " - time " << current_time << " - dt " << dt << " ###" << std::endl;

        // Update the surrogate boundary Dirichlet value from the previous time step velocity gradient
        SbmUtilities<dim>::UpdateSurrogateBoundaryDirichletValues(mass_factor, box_divisions, cell_size, surrogate_cells, surrogate_nodes, lumped_mass_vector, distance_vects, v, v_surrogate);
        for (unsigned int i = 0; i < num_nodes; ++i) {
            if (surrogate_nodes[i]) {
                for (unsigned int d = 0; d < dim; ++d) {
                    v(i, d) = v_surrogate(i, d);
                    v_n(i, d) = v_surrogate(i, d);
                }
            }
        }

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

            for (unsigned int i = 0; i < num_nodes; ++i) {
                for (unsigned int d = 0; d < dim; ++d) {
                    rk_v(i, d) += v_n(i, d);
                }
            }

            for (unsigned int i_dof = 0; i_dof < n_fixed_dofs; ++i_dof) {
                const unsigned int dof_row = fixed_dofs_rows[i_dof];
                const unsigned int dof_col = fixed_dofs_cols[i_dof];
                rk_v(dof_row, dof_col) = rk_theta * v(dof_row, dof_col) + (1.0 - rk_theta) * v_n(dof_row, dof_col); // Set BC value in fixed DOFs
            }

            // Calculate current step residual
            if constexpr (dim == 2) {
                std::array<int,4> cell_node_ids;
                Eigen::Array<double, 4, 2> cell_v;
                Eigen::Array<double, 4, 2> cell_f;
                Eigen::Array<double, 4, 2> cell_acc;
                Eigen::Array<double, 8, 1> cell_res;
                for (unsigned int i = 0; i < box_divisions[1]; ++i) {
                    for (unsigned int j = 0; j < box_divisions[0]; ++j) {
                        const unsigned int i_cell = CellUtilities::GetCellGlobalId(i, j, box_divisions);
                        if (active_cells[i_cell]) {
                            // Get current cell data
                            CellUtilities::GetCellNodesGlobalIds(i, j, box_divisions, cell_node_ids);
                            const double cell_p = p[i_cell];
                            cell_v = rk_v(cell_node_ids, Eigen::all);

                            for (unsigned int i_node = 0; i_node < cell_nodes; ++i_node) {
                                for (unsigned int d = 0; d < dim; ++d) {
                                    cell_f(i_node, d) = f(cell_node_ids[i_node], d);
                                    cell_acc(i_node, d) = acc(cell_node_ids[i_node], d);
                                }
                            }

                            // Calculate current cell residual
                            IncompressibleNavierStokesQ1P0StructuredElement::CalculateRightHandSide(cell_size[0], cell_size[1], mu, rho, cell_v, cell_p, cell_f, cell_acc, cell_res);

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
        for (unsigned int i_dof = 0; i_dof < n_free_dofs; ++i_dof) {
            const unsigned int dof_row = free_dofs_rows[i_dof];
            const unsigned int dof_col = free_dofs_cols[i_dof];
            v(dof_row, dof_col) = 0.0;
            for (unsigned int rk_step = 0; rk_step < rk_num_steps; ++rk_step) {
                v(dof_row, dof_col) += rk_B[rk_step] * rk_res[rk_step](dof_row, dof_col);
            }
            v(dof_row, dof_col) *= dt * lumped_mass_vector_inv(dof_row, dof_col);
            v(dof_row, dof_col) += v_n(dof_row, dof_col);
        }
        std::cout << "Velocity prediction solved." << std::endl;

        // Solve pressure update
        Operators<dim>::ApplyDivergenceOperator(box_divisions, cell_size, active_cells, v, delta_p_rhs);
        for (unsigned int i = 0; i < num_cells; ++i) {
            delta_p_rhs[i] = -delta_p_rhs[i] / dt;
        }

        const bool is_converged = cg.Solve(delta_p_rhs, delta_p);
        for (unsigned int i = 0; i < num_cells; ++i) {
            p[i] += delta_p[i];
        }
        tot_p_iters += cg.Iterations();
        if (is_converged) {
            std::cout << "Pressure problem converged in " << cg.Iterations() << " iterations." << std::endl;
        } else {
            std::cout << "Pressure problem did not converge in " << cg.Iterations() << " iterations." << std::endl;
        }

        // Correct velocity
        Operators<dim>::ApplyGradientOperator(box_divisions, cell_size, active_cells, delta_p, delta_p_grad);
        for (unsigned int i_dof = 0; i_dof < n_free_dofs; ++i_dof) {
            const unsigned int dof_row = free_dofs_rows[i_dof];
            const unsigned int dof_col = free_dofs_cols[i_dof];
            v(dof_row, dof_col) += dt * lumped_mass_vector_inv(dof_row, dof_col) * delta_p_grad(dof_row, dof_col);
        }
        std::cout << "Velocity update finished." << std::endl;

        // Output current step solution
        if ((current_step % output_interval) < 1.0e-12)  {
            const static Eigen::IOFormat out_format(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
            std::ofstream v_file(results_path + "v_" + std::to_string(current_step) + ".txt");
            std::ofstream p_file(results_path + "p_" + std::to_string(current_step) + ".txt");
            if (v_file.is_open()) {
                v_file << current_time << std::endl;
                for (unsigned int i = 0; i < num_nodes; ++i) {
                    for (unsigned int d = 0; d < dim; ++d) {
                        v_file << v(i,d);
                        if (d < dim - 1) {
                            v_file << ", ";
                        } else {
                            v_file << "\n";
                        }
                    }
                }
                v_file.close();
            }
            if (p_file.is_open()) {
                p_file << current_time << std::endl;
                for (unsigned int i = 0; i < num_cells; ++i) {
                    p_file << p[i] << std::endl;
                }
                p_file.close();
            }
            std::cout << "Results output completed." << std::endl;
        }

        // Update variables for next time step
        for (unsigned int i = 0; i < num_nodes; ++i) {
            for (unsigned int d = 0; d < dim; ++d) {
                acc(i, d) = (v(i, d) - v_n(i, d)) / dt;
                v_n(i, d) = v(i, d);
            }
        }
        ++current_step;
        current_time += dt;
    }

    // Print final data
    std::cout << "TOTAL PRESSURE ITERATIONS: " << tot_p_iters << std::endl;

    return 0;
}