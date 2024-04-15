#include <iostream>

#include "runge_kutta_utilities.hpp"

template<>
void RungeKuttaUtilities<4>::SetNodesVector(std::array<double, 4>& rNodesVector)
{
    rNodesVector[0] = 0.0;
    rNodesVector[1] = 0.5;
    rNodesVector[2] = 0.5;
    rNodesVector[3] = 1.0;
}

template<>
void RungeKuttaUtilities<4>::SetWeightsVector(std::array<double, 4>& rWeighsVector)
{
    rWeighsVector[0] = 1.0 / 6.0;
    rWeighsVector[1] = 1.0 / 3.0;
    rWeighsVector[2] = 1.0 / 3.0;
    rWeighsVector[3] = 1.0 / 6.0;
}

template <>
void RungeKuttaUtilities<4>::SetRungeKuttaMatrix(Eigen::Array<double, 4, 4>& rRungeKuttaMatrix)
{
    rRungeKuttaMatrix.setZero();
    rRungeKuttaMatrix(1, 0) = 0.5;
    rRungeKuttaMatrix(2, 1) = 0.5;
    rRungeKuttaMatrix(3, 2) = 1.0;
}

template class RungeKuttaUtilities<4>;
