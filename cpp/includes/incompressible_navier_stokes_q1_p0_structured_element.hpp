#include <Eigen/Dense>

#pragma once

class IncompressibleNavierStokesQ1P0StructuredElement
{
public:

    static void CalculateRightHandSide(
        const double a,
        const double b,
        const double mu,
        const double rho,
        const Eigen::Array<double, 4, 2>& v,
        const double p,
        const Eigen::Array<double, 4, 2>& f,
        const Eigen::Array<double, 4, 2>& acc,
        Eigen::Array<double, 8, 1>& RHS);

    static void CalculateRightHandSide(
        const double a,
        const double b,
        const double c,
        const double mu,
        const double rho,
        const Eigen::Array<double, 8, 3>& v,
        const double p,
        const Eigen::Array<double, 8, 3>& f,
        const Eigen::Array<double, 8, 3>& acc,
        Eigen::Array<double, 24, 1>& RHS);

    static void GetCellGradientOperator(
        const double a,
        const double b,
        Eigen::Array<double, 4, 2>& G);

    static void GetCellGradientOperator(
        const double a,
        const double b,
        const double c,
        Eigen::Array<double, 8, 3>& G);
};