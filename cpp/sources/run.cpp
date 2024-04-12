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
#include <Eigen/IterativeLinearSolvers>
#include <unsupported/Eigen/FFT>

#include "cell_utilities.hpp"
#include "incompressible_navier_stokes_q1_p0_structured_element.hpp"
#include "matrix_replacement.hpp"
#include "mesh_utilities.hpp"
#include "operators.hpp"
#include "pressure_preconditioner.hpp"
#include "pressure_conjugate_gradient_solver.hpp"
#include "runge_kutta_utilities.hpp"
#include "sbm_utilities.hpp"
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
    const std::array<int, dim> box_divisions({150, 30});
    // const std::array<int, dim> box_divisions({3, 3});

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
    std::cout << "\n### OUTPUT FOLDER ###" << std::endl;
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
    Eigen::Array<double, Eigen::Dynamic, dim> nodal_coords;
    MeshUtilities<dim>::CalculateNodalCoordinates(box_size, box_divisions, nodal_coords);

    // Write coordinates and connectivities for postprocess
    std::ofstream coordinates_file(results_path + "coordinates.txt");
    if (coordinates_file.is_open()) {
        for (unsigned int i = 0; i < num_nodes; ++i) {
            const auto& r_coords = nodal_coords.row(i);
            if constexpr (dim == 2) {
                coordinates_file << r_coords(0) << " " << r_coords(1) << " " << 0.0 << std::endl;
            } else {
                coordinates_file << r_coords(0) << " " << r_coords(1) << " " << r_coords(2) << std::endl;
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
    Eigen::ArrayXd p = Eigen::VectorXd::Zero(num_cells);
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
        if (r_coords[1] > (1.0-tol)) {
            fixity(i_node, 1) = true; // y-velocity
        }

        // Bottom wall
        if (r_coords[1] < tol) {
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
        // distance(i_node) = 1.0;
        // distance(i_node) = r_coords[1] - 0.5;
    }

    std::ofstream distance_file(results_path + "distance.txt");
    if (distance_file.is_open()) {
        for (unsigned int i = 0; i < num_nodes; ++i) {
            distance_file << distance(i) << std::endl;
        }
    }

    // Define the surrogate boundary
    std::cout << "\n### SURROGATE BOUNDARY DEFINITION ###" << std::endl;
    Eigen::Array<double, Eigen::Dynamic, dim> v_surrogate(num_nodes, dim);
    v_surrogate.setZero();

    Eigen::Array<bool, Eigen::Dynamic, 1> surrogate_nodes(num_nodes);
    SbmUtilities<dim>::FindSurrogateBoundaryNodes(box_divisions, distance, surrogate_nodes);
    std::cout << "Surrogate boundary nodes found." << std::endl;

    Eigen::Array<bool, Eigen::Dynamic, 1> surrogate_cells(num_cells);
    SbmUtilities<dim>::FindSurrogateBoundaryCells(box_divisions, distance, surrogate_nodes, surrogate_cells);
    std::cout << "Surrogate boundary cells found." << std::endl;

    // Calculate the distance vectors in the surrogate boundary nodes
    Eigen::Vector<double, dim> aux_dir;
    Eigen::Array<double, Eigen::Dynamic, dim> distance_vects(num_nodes, dim);
    for (unsigned int i_node = 0; i_node < num_nodes; ++i_node) {
        if (surrogate_nodes(i_node)) {
            const auto& r_coords = nodal_coords.row(i_node);
            aux_dir = cyl_orig - r_coords;
            aux_dir /= aux_dir.norm();
            distance_vects.row(i_node) = distance[i_node] * aux_dir;
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
        if (surrogate_nodes(i_node)) {
            fixity.row(i_node).setConstant(true);
        }
    }

    // Deactivate the interior and intersected cells (SBM-like resolution)
    Eigen::Array<bool, Eigen::Dynamic, 1> active_cells(num_cells);
    std::vector<bool> active_cells_vect(num_cells);
    if constexpr (dim == 2) {
        std::array<int,4> cell_node_ids;
        for (unsigned int i = 0; i < box_divisions[1]; ++i) {
            for (unsigned int j = 0; j < box_divisions[0]; ++j) {
                CellUtilities::GetCellNodesGlobalIds(i, j, box_divisions, cell_node_ids);
                const auto cell_fixity = fixity(cell_node_ids, Eigen::all);
                active_cells(CellUtilities::GetCellGlobalId(i, j, box_divisions)) = !cell_fixity.all();
                active_cells_vect[CellUtilities::GetCellGlobalId(i, j, box_divisions)] = !cell_fixity.all();
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
    Eigen::ArrayXd fft_c(num_cells);
    PressurePreconditioner pressure_precond;
    auto free_cell_result = MeshUtilities<dim>::FindFirstFreeCellId(box_divisions, fixity);
    if (std::get<0>(free_cell_result)) {
        const unsigned int free_cell_id = std::get<1>(free_cell_result);
        std::cout << "Free cell id: " << free_cell_id << "." <<std::endl;
        Eigen::VectorXd x(num_cells);
        Eigen::VectorXd y(num_cells);
        x.setZero();
        x(free_cell_id) = 1.0;
        Operators<dim>::ApplyPressureOperator(box_divisions, cell_size, active_cells, lumped_mass_vector_inv, x, y);

        Eigen::Matrix<std::complex<double>, Eigen::Dynamic, 1> fft_x(num_cells); // Complex array for FFT(x) output
        Eigen::Matrix<std::complex<double>, Eigen::Dynamic, 1> fft_y(num_cells); // Complex array for FFT(y) output
        Eigen::Matrix<std::complex<double>, Eigen::Dynamic, 1> x_complex(num_cells, 1); // Complex array for FFT(x) input
        Eigen::Matrix<std::complex<double>, Eigen::Dynamic, 1> y_complex(num_cells, 1); // Complex array for FFT(y) input
        for (unsigned int i = 0; i < num_cells; ++i) {
            x_complex(i) = x(i); // Set x_complex real data from x
            y_complex(i) = y(i); // Set y_complex real data from y
        }

        Eigen::FFT<double> fft;
        fft.fwd(fft_x, x_complex);
        fft.fwd(fft_y, y_complex);
        fft_c = (fft_y.array() / fft_x.array()).real(); // Take the real part only (imaginary one is zero)
        fft_c(0) = 1.0; // Remove the first coefficient as this is associated to the solution average
        pressure_precond.setFFT(fft_c); // Instantiate the pressure preconditioner
        std::cout << "Pressure preconditioner set." << std::endl;
    } else {
        std::cout << "There is no cell with all the DOFs free. No pressure preconditioner can be set." << std::endl;
    }

    // Set Runge-Kutta arrays
    constexpr int rk_order = 4;
    Eigen::Array<double, rk_order, 1> rk_B;
    Eigen::Array<double, rk_order, 1> rk_C;
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
    // Eigen::VectorXd delta_p(num_cells);
    Eigen::Array<double, Eigen::Dynamic, 1> delta_p(num_cells);
    delta_p.setZero();
    Eigen::VectorXd delta_p_rhs(num_cells);
    MatrixReplacement<dim> matrix_replacement(box_divisions, cell_size, active_cells, lumped_mass_vector_inv_bcs);

    // Allocate auxiliary arrays for the velocity update
    Eigen::Matrix<double, Eigen::Dynamic, dim> delta_p_grad(num_nodes, dim);

    const double abs_tol = 1.0e-15;
    const double rel_tol = 1.0e-15;
    const double max_iter = 200;
    PressureOperator<dim> pressure_operator(box_divisions, cell_size, active_cells_vect, lumped_mass_vector_inv_bcs);
    PressureConjugateGradientSolver<dim> cg(abs_tol, rel_tol, max_iter, pressure_operator, fft_c);


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
        //TODO: We can get rid of v_surrogate (but lets keep it for debugging for a while)
        for (unsigned int i = 0; i < num_nodes; ++i) {
            if (surrogate_nodes(i)) {
                v(i, Eigen::all) = v_surrogate.row(i);
                v_n(i, Eigen::all) = v_surrogate.row(i);
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
            rk_v += v_n;

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
                        if (active_cells(i_cell)) {
                            // Get current cell data
                            CellUtilities::GetCellNodesGlobalIds(i, j, box_divisions, cell_node_ids);
                            const double cell_p = p(i_cell);
                            cell_v = rk_v(cell_node_ids, Eigen::all);
                            cell_f = f(cell_node_ids, Eigen::all);
                            cell_acc = acc(cell_node_ids, Eigen::all);

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
            delta_p_rhs(i) = -delta_p_rhs(i) / dt;
        }

        // Eigen::ConjugateGradient<MatrixReplacement<dim>, Eigen::Lower | Eigen::Upper, Eigen::IdentityPreconditioner> cg;
        // // Eigen::ConjugateGradient<MatrixReplacement<dim>, Eigen::Lower|Eigen::Upper, PressurePreconditioner> cg;
        // // cg.preconditioner() = pressure_precond;
        // cg.compute(matrix_replacement);
        // delta_p = cg.solve(delta_p_rhs);
        // p += delta_p.array();
        // tot_p_iters += cg.iterations();
        // std::cout << "Pressure problem solved in " << cg.iterations() << " iterations." << std::endl;

        cg.Solve(delta_p_rhs, delta_p);
        p += delta_p;
        tot_p_iters += cg.Iterations();
        std::cout << "Pressure problem solved in " << cg.Iterations() << " iterations." << std::endl;

        // Correct velocity
        Operators<dim>::ApplyGradientOperator(box_divisions, cell_size, active_cells, delta_p, delta_p_grad);
        for (unsigned int i_dof = 0; i_dof < n_free_dofs; ++i_dof) {
            const unsigned int dof_row = free_dofs_rows[i_dof];
            const unsigned int dof_col = free_dofs_cols[i_dof];
            v(dof_row, dof_col) += dt * lumped_mass_vector_inv(dof_row, dof_col) * delta_p_grad(dof_row, dof_col);
        }
        std::cout << "Velocity update finished." << std::endl;

        // Output current step solution
        const static Eigen::IOFormat out_format(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
        std::ofstream v_file(results_path + "v_" + std::to_string(current_step) + ".txt");
        std::ofstream p_file(results_path + "p_" + std::to_string(current_step) + ".txt");
        if (v_file.is_open()) {
            v_file << current_time << std::endl;
            v_file << v.format(out_format);
            v_file.close();
        }
        if (p_file.is_open()) {
            p_file << current_time << std::endl;
            p_file << p.format(out_format);
            p_file.close();
        }
        std::cout << "Results output completed.\n" << std::endl;

        // Update variables for next time step
        acc = (v - v_n) / dt;
        v_n = v;
        ++current_step;
        current_time += dt;
    }

    // Print final data
    std::cout << "TOTAL PRESSURE ITERATIONS: " << tot_p_iters << std::endl;

    return 0;
}