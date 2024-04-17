#include <array>
#include <utility>
#include <vector>
#include "include/experimental/mdspan"

#pragma once

template<int TOrder>
class RungeKuttaUtilities
{
public:

    using RungeKuttaVector = std::array<double, TOrder>;

    using RungeKuttaMatrixExtent = std::experimental::extents<std::size_t, TOrder, TOrder>;

    using RungeKuttaMatrixView = std::experimental::mdspan<double, RungeKuttaMatrixExtent>;

    static void SetNodesVector(RungeKuttaVector& rNodesVector);

    static void SetWeightsVector(RungeKuttaVector& rWeightsVector);

    static void SetRungeKuttaMatrix(RungeKuttaMatrixView& rRungeKuttaMatrix);

};