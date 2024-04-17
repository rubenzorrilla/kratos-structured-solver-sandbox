#include <Eigen/Dense>
#include "include/experimental/mdspan"

#pragma once

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

    static void GetCellGradientOperator(
        const double a,
        const double b,
        Eigen::Array<double, 4, 2>& G);

    static void GetCellGradientOperator(
        const double a,
        const double b,
        QuadVectorDataView& G);

    static void GetCellGradientOperator(
        const double a,
        const double b,
        const double c,
        Eigen::Array<double, 8, 3>& G);

    static void GetCellGradientOperator(
        const double a,
        const double b,
        const double c,
        HexaVectorDataView& G);
};