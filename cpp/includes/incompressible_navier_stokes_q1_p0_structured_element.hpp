#pragma once

// #include <Eigen/Dense>
#include "include/experimental/mdspan"

// Intel sycl
#include <CL/sycl.hpp>

class IncompressibleNavierStokesQ1P0StructuredElement
{
public:

    using QuadVectorDataView = std::experimental::mdspan<double, std::experimental::extents<std::size_t, 4, 2>>;

    using HexaVectorDataView = std::experimental::mdspan<double, std::experimental::extents<std::size_t, 8, 3>>;

    static void CalculateRightHandSide(
        const double a,
        const double b,
        const double mu,
        const double rho,
        const QuadVectorDataView& v,
        const double p,
        const QuadVectorDataView& f,
        const QuadVectorDataView& acc,
        std::array<double, 8>& RHS);

    static void CalculateRightHandSide(
        const double a,
        const double b,
        const double c,
        const double mu,
        const double rho,
        const HexaVectorDataView& v,
        const double p,
        const HexaVectorDataView& f,
        const HexaVectorDataView& acc,
        std::array<double, 24>& RHS);

    // static void GetCellGradientOperator(
    //     const double a,
    //     const double b,
    //     Eigen::Array<double, 4, 2>& G);

     SYCL_EXTERNAL static void GetCellGradientOperator(
        const double a,
        const double b,
        QuadVectorDataView& G) 
    {
        const double cG0 = 0.5*b;
        const double cG1 = -cG0;
        const double cG2 = 0.5*a;
        const double cG3 = -cG2;
        G(0,0) = cG1;
        G(0,1) = cG3;
        G(1,0) = cG0;
        G(1,1) = cG3;
        G(2,0) = cG0;
        G(2,1) = cG2;
        G(3,0) = cG1;
        G(3,1) = cG2;
    }

    // static void GetCellGradientOperator(
    //     const double a,
    //     const double b,
    //     const double c,
    //     Eigen::Array<double, 8, 3>& G);

     SYCL_EXTERNAL static void GetCellGradientOperator(
        const double a,
        const double b,
        const double c,
        HexaVectorDataView& G) 
    {    
        //substitute_G_3d
    }
};