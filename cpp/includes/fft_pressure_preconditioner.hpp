#include <cmath>
#include <array>
#include <memory>
#include <vector>
#include <utility>
#include <fftw3.h>

#include "include/experimental/mdspan"

#include "pressure_preconditioner.hpp"
#include "mesh_utilities.hpp"
#include "operators.hpp"
#include "pressure_operator.hpp"

#pragma once

template<int TDim>
class FftPressurePreconditioner : public PressurePreconditioner<TDim>
{
public:

    using BaseType = PressurePreconditioner<TDim>;

    using SizeType = BaseType::SizeType;

    using IndexType = BaseType::IndexType;

    using VectorType = BaseType::VectorType;

    using FixityViewType = std::experimental::mdspan<bool, std::experimental::extents<std::size_t, std::dynamic_extent, TDim>>;

    using MatrixViewType = std::experimental::mdspan<double, std::experimental::extents<std::size_t, std::dynamic_extent, TDim>>;

    FftPressurePreconditioner()
        : PressurePreconditioner<TDim>()
    {
    }

    FftPressurePreconditioner(
        const std::array<int, TDim>& rBoxDivisions,
        const std::array<double, TDim>& rCellSize,
        const FixityViewType& rFixity,
        const double MassFactor)
        : PressurePreconditioner<TDim>()
        , mrBoxDivisions(rBoxDivisions)
        , mrCellSize(rCellSize)
        , mrFixity(rFixity)
        , mMassFactor(MassFactor)
    {
    }

    /**
     * @brief Create the preconditioner for the pressure CG solver
     * For this we convert the periodic pressure matrix C (with no velocity BCs) to FFT
     * The simplest way is to generate any vector x, compute the image vector y=C*X
     * The transform of C is fft(y)./fft(x), as C is cyclic the transformed C must be diagonal
     * The resulting transformed coefficientes should be real, because the operator is symmetric.
     * Also it should be semidefinite positive (SPD) because the Laplacian operator is SPD
     * The first coefficient is null, because the Laplacian is not PD, just SPD.
     * But we can replace this null coefficient by anything different from 0.
     * At most it would degrade the convergence of the PCG, but we will see that the convergence is OK.
     */
    void SetUp() override
    {
        // Search for the first free cell to obtain the periodic (circulant) stencil from it
        auto free_cell_result = MeshUtilities<TDim>::FindFirstFreeCellId(mrBoxDivisions, mrFixity);
        if (std::get<0>(free_cell_result)) {
            // Memory allocation
            auto mesh_data = MeshUtilities<TDim>::CalculateMeshData(mrBoxDivisions);
            const SizeType num_nodes = std::get<0>(mesh_data);
            mProblemSize = std::get<1>(mesh_data);
            mFFTc.resize(mProblemSize);

            // Build the periodic (i.e. no BCs) lumped mass matrix inverse
            double periodic_lumped_mass_data_inv[num_nodes * TDim];
            MatrixViewType periodic_lumped_mass_vector_inv(periodic_lumped_mass_data_inv, num_nodes, TDim);
            MeshUtilities<TDim>::CalculateLumpedMassVector(mMassFactor, mrBoxDivisions, periodic_lumped_mass_vector_inv);
            for (IndexType i_node = 0; i_node < num_nodes; ++i_node) {
                for (IndexType d = 0; d < TDim; ++d) {
                    periodic_lumped_mass_vector_inv(i_node, d) = 1.0 / periodic_lumped_mass_vector_inv(i_node, d);
                }
            }

            // Get the circulant row from the pressure operator application onto a vector
            // Note that we set the vector such that we get the row corresponding to the previously obtained free cell
            const IndexType free_cell_id = std::get<1>(free_cell_result);
            std::cout << "Free cell id: " << free_cell_id << "." <<std::endl;
            VectorType x(mProblemSize, 0.0);
            VectorType y(mProblemSize);
            x[free_cell_id] = 1.0;

            PressureOperator<TDim> periodic_pressure_operator(mrBoxDivisions, mrCellSize, periodic_lumped_mass_vector_inv);
            periodic_pressure_operator.Apply(x, y);
            //periodic_pressure_operator.Output("pressure_matrix_without_bcs_" + std::to_string(mrBoxDivisions[0]) + "_" + std::to_string(mrBoxDivisions[1]), "../cpp_output/");

            // std::unique_ptr<PressureOperator<dim>> p_pressure_operator;
            // if (artificial_compressibility) {
            //     auto p_aux = std::make_unique<PressureOperator<dim>>(rho, sound_velocity, box_divisions, cell_size, active_cells, periodic_lumped_mass_vector_inv);
            //     std::swap(p_pressure_operator, p_aux);
            // } else {
            //     auto p_aux = std::make_unique<PressureOperator<dim>>(box_divisions, cell_size, active_cells, periodic_lumped_mass_vector_inv);
            //     std::swap(p_pressure_operator, p_aux);
            // }
            // p_pressure_operator->Apply(x, y);

            // Allocate memory for the FFTW transforms
            // TODO: Use real numbers in here
            fftw_complex *fft_x;
            fftw_complex *fft_y;
            fftw_complex *x_complex;
            fftw_complex *y_complex;
            fft_x = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * mProblemSize);
            fft_y = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * mProblemSize);
            x_complex = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * mProblemSize);
            y_complex = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * mProblemSize);

            // Create the FFTW fransform plans
            fftw_plan p_x;
            fftw_plan p_y;
            p_x = fftw_plan_dft(TDim, mrBoxDivisions.data(), x_complex, fft_x, FFTW_FORWARD, FFTW_ESTIMATE); //TODO: We should use the FFT for real numbers in here
            p_y = fftw_plan_dft(TDim, mrBoxDivisions.data(), y_complex, fft_y, FFTW_FORWARD, FFTW_ESTIMATE); //TODO: We should use the FFT for real numbers in here

            // Set FFTs input data
            // Note that FFTW advises to do this after setting the plans as some of the flags may overwrite this
            for (IndexType i = 0; i < mProblemSize; ++i) {
                x_complex[i][0] = x[i]; // Setting real part
                x_complex[i][1] = 0.0; // Setting imaginary part
                y_complex[i][0] = y[i]; // Setting real part
                y_complex[i][1] = 0.0; // Setting imaginary part
            }

            // Execute the FFTs
            fftw_execute(p_x);
            fftw_execute(p_y);

            // Avoid the zero eigenvalue modes
            const double tol = 1.0e-15;
            for (IndexType i = 0; i < mProblemSize; ++i) {
                const double num = fft_y[i][0] * fft_x[i][0] + fft_y[i][1] * fft_x[i][1];
                const double den = std::pow(fft_x[i][0], 2) + std::pow(fft_x[i][1], 2);
                const double real_part = num / den; // Take the real part only (imaginary one is zero)
                if (std::abs(real_part) < tol) {
                    std::cout << "Rigid body mode found in component: " << i << std::endl;
                    mFFTc[i] = 1.0;
                } else {
                    mFFTc[i] = real_part;
                }
            }

            // Destroy FFTW transform plans and free memory
            fftw_destroy_plan(p_x);
            fftw_destroy_plan(p_y);
            fftw_free(fft_x);
            fftw_free(fft_y);
            fftw_free(x_complex);
            fftw_free(y_complex);

            // Allocate memory for the preconditioner application (note that this is done once)
            // TODO: Use real numbers in here
            mFFTb = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * mProblemSize);
            mComplexb = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * mProblemSize);

            mIFFTAux = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * mProblemSize);
            mComplexAux = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * mProblemSize);

            // Establish the FFTW transform plans for the preconditioner application
            mPlanForward = fftw_plan_dft(TDim, mrBoxDivisions.data(), mComplexb, mFFTb, FFTW_FORWARD, FFTW_MEASURE);
            mPlanBackward = fftw_plan_dft(TDim, mrBoxDivisions.data(), mComplexAux, mIFFTAux, FFTW_BACKWARD, FFTW_MEASURE);

            // Set the initialization flag
            mIsSetUp = true;
            std::cout << "Pressure preconditioner set." << std::endl;
        } else {
            std::cout << "There is no cell with all the DOFs free. No pressure preconditioner can be set." << std::endl;
        }
    }

    void Apply(
        const VectorType& rInput,
        VectorType& rOutput) override
    {
        if (mIsSetUp) {
            // Set the forward FFT input
            for (unsigned int i = 0; i < mProblemSize; ++i) {
                mComplexb[i][0] = rInput[i]; // Setting real part
                mComplexb[i][1] = 0.0; // Setting imaginary part
            }

            // Execute the forward FFT
            fftw_execute(mPlanForward);

            // Set the backward FFT input
            for (unsigned int i = 0; i < mProblemSize; ++i) {
                const double num_real = mFFTb[i][0] * mFFTc[i]; // Note that in here we are assuming that the imaginary part of mFFTc is zero
                const double num_imag = mFFTb[i][1] * mFFTc[i]; // Note that in here we are assuming that the imaginary part of mFFTc is zero
                const double den = std::pow(mFFTc[i], 2); // Note that in here we are assuming that the imaginary part of mFFTc is zero
                mComplexAux[i][0] = num_real / den;
                mComplexAux[i][1] = num_imag / den;
            }

            // Execute the backward FFT
            fftw_execute(mPlanBackward);

            // Set the output as the normalized IFFT
            // Note from FFTW documentation "FFTW computes an unnormalized DFT.Thus, computing a forward
            // followed by a backward transform (or vice versa) results in the original array scaled by n."
            for (unsigned int i = 0; i < mProblemSize; ++i) {
                rOutput[i] = mIFFTAux[i][0] / mProblemSize;
            }
        } else {
            std::cout << "Pressure preconditioner is not set. Please call the 'SetUp' method first." << std::endl;
        }
    }

    void Clear() override
    {
        mIsSetUp = false;

        mProblemSize = 0;

        mFFTc.clear();

        fftw_free(mFFTb);
        fftw_free(mComplexb);
        fftw_free(mIFFTAux);
        fftw_free(mComplexAux);

        fftw_destroy_plan(mPlanForward);
        fftw_destroy_plan(mPlanBackward);
    }

private:

    const std::array<int, TDim>& mrBoxDivisions;

    const std::array<double, TDim>& mrCellSize;

    const FixityViewType& mrFixity;

    const double mMassFactor;

    bool mIsSetUp = false;

    SizeType mProblemSize;

    VectorType mFFTc;

    fftw_complex *mFFTb;

    fftw_complex *mComplexb;

    fftw_complex *mIFFTAux;

    fftw_complex *mComplexAux;

    fftw_plan mPlanForward;

    fftw_plan mPlanBackward;

};

    //TODO: This is the old implementation (keep it in here for a while just in case)

    // std::vector<double> fft_c(num_cells);
    // auto free_cell_result = MeshUtilities<dim>::FindFirstFreeCellId(box_divisions, fixity);
    // if (std::get<0>(free_cell_result)) {
    //     double periodic_lumped_mass_data_inv[num_nodes * dim];
    //     MatrixViewType periodic_lumped_mass_vector_inv(periodic_lumped_mass_data_inv, num_nodes, dim);
    //     MeshUtilities<dim>::CalculateLumpedMassVector(mass_factor, box_divisions, periodic_lumped_mass_vector_inv);
    //     for (unsigned int i_node = 0; i_node < num_nodes; ++i_node) {
    //         for (unsigned int d = 0; d < dim; ++d) {
    //             periodic_lumped_mass_vector_inv(i_node, d) = 1.0 / periodic_lumped_mass_vector_inv(i_node, d);
    //         }
    //     }

    //     const unsigned int free_cell_id = std::get<1>(free_cell_result);
    //     std::cout << "Free cell id: " << free_cell_id << "." <<std::endl;
    //     std::vector<double> x(num_cells, 0.0);
    //     std::vector<double> y(num_cells);
    //     x[free_cell_id] = 1.0;

    //     PressureOperator<dim> periodic_pressure_operator(box_divisions, cell_size, periodic_lumped_mass_vector_inv);
    //     periodic_pressure_operator.Apply(x, y);
    //     // periodic_pressure_operator.f("pressure_matrix_without_bcs" + std::to_string(box_divisions[0]) + "_" + std::to_string(box_divisions[1]), results_path);

    //     // std::unique_ptr<PressureOperator<dim>> p_pressure_operator;
    //     // if (artificial_compressibility) {
    //     //     auto p_aux = std::make_unique<PressureOperator<dim>>(rho, sound_velocity, box_divisions, cell_size, active_cells, periodic_lumped_mass_vector_inv);
    //     //     std::swap(p_pressure_operator, p_aux);
    //     // } else {
    //     //     auto p_aux = std::make_unique<PressureOperator<dim>>(box_divisions, cell_size, active_cells, periodic_lumped_mass_vector_inv);
    //     //     std::swap(p_pressure_operator, p_aux);
    //     // }
    //     // p_pressure_operator->Apply(x, y);

    //     // std::vector<std::complex<double>> fft_x(num_cells); // Complex array for FFT(x) output
    //     // std::vector<std::complex<double>> fft_y(num_cells); // Complex array for FFT(y) output
    //     // std::vector<std::complex<double>> x_complex(num_cells); // Complex array for FFT(x) input
    //     // std::vector<std::complex<double>> y_complex(num_cells); // Complex array for FFT(y) input
    //     // for (unsigned int i = 0; i < num_cells; ++i) {
    //     //     x_complex[i].real(x[i]); // Set x_complex real data from x
    //     //     y_complex[i].real(y[i]); // Set y_complex real data from y
    //     // }

    //     // Eigen::FFT<double> fft;
    //     // fft.fwd(fft_x, x_complex);
    //     // fft.fwd(fft_y, y_complex);
    //     // for (unsigned int i = 0; i < num_cells; ++i) {
    //     //     fft_c[i] = (fft_y[i] / fft_x[i]).real(); // Take the real part only (imaginary one is zero)
    //     // }
    //     // fft_c[0] = 1.0; // Remove the first coefficient as this is associated to the solution average

    //     fftw_complex *fft_x_new;
    //     fftw_complex *fft_y_new;
    //     fftw_complex *x_complex_new;
    //     fftw_complex *y_complex_new;
    //     fft_x_new = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * num_cells);
    //     fft_y_new = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * num_cells);
    //     x_complex_new = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * num_cells);
    //     y_complex_new = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * num_cells);

    //     //TODO: We should use the FFT for real numbers in here

    //     fftw_plan p_x;
    //     fftw_plan p_y;
    //     p_x = fftw_plan_dft(dim, box_divisions.data(), x_complex_new, fft_x_new, FFTW_FORWARD, FFTW_ESTIMATE);
    //     p_y = fftw_plan_dft(dim, box_divisions.data(), y_complex_new, fft_y_new, FFTW_FORWARD, FFTW_ESTIMATE);

    //     for (unsigned int i = 0; i < num_cells; ++i) {
    //         x_complex_new[i][0] = x[i]; // Setting real part
    //         x_complex_new[i][1] = 0.0; // Setting imaginary part
    //         y_complex_new[i][0] = y[i]; // Setting real part
    //         y_complex_new[i][1] = 0.0; // Setting imaginary part
    //     }

    //     fftw_execute(p_x);
    //     fftw_execute(p_y);

    //     const double tol = 1.0e-15;
    //     for (unsigned int i = 0; i < num_cells; ++i) {
    //         const double num = fft_y_new[i][0] * fft_x_new[i][0] + fft_y_new[i][1] * fft_x_new[i][1];
    //         const double den = std::pow(fft_x_new[i][0], 2) + std::pow(fft_x_new[i][1], 2);
    //         const double real_part = num / den; // Take the real part only (imaginary one is zero)
    //         if (std::abs(real_part) < tol) {
    //             std::cout << "Rigid body mode found in component: " << i << std::endl;
    //             fft_c[i] = 1.0;
    //         } else {
    //             fft_c[i] = real_part;
    //         }
    //     }

    //     fftw_destroy_plan(p_x);
    //     fftw_destroy_plan(p_y);

    //     std::cout << "Pressure preconditioner set." << std::endl;
    // } else {
    //     std::cout << "There is no cell with all the DOFs free. No pressure preconditioner can be set." << std::endl;
    // }