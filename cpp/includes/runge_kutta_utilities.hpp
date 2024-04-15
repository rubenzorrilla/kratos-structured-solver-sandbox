#include <array>
#include <utility>
#include <vector>
#include <Eigen/Dense>

#pragma once

template<int TOrder>
class RungeKuttaUtilities
{
public:

    static void SetNodesVector(std::array<double, TOrder>& rNodesVector);

    static void SetWeightsVector(std::array<double, TOrder>& rWeightsVector);

    static void SetRungeKuttaMatrix(Eigen::Array<double, TOrder, TOrder>& rRungeKuttaMatrix);

};