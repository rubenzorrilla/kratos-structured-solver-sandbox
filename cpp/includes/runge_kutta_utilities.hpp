#include <array>
#include <utility>
#include <vector>
#include <Eigen/Dense>

#pragma once

template<int TOrder>
class RungeKuttaUtilities
{
public:

    static void SetNodesVector(Eigen::Array<double, TOrder, 1>& rNodesVector);

    static void SetWeightsVector(Eigen::Array<double, TOrder, 1>& rWeightsVector);

    static void SetRungeKuttaMatrix(Eigen::Array<double, TOrder, TOrder>& rRungeKuttaMatrix);

};