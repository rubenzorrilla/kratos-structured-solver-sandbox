#include <array>
// #include <Eigen/Dense>
#include "include/experimental/mdspan"

// Intel sycl
#include <CL/sycl.hpp> 

#pragma once

template<int TDim, class src_accessor_t = std::vector<double>, class dst_accessor_t = std::vector<double>>
class Operators
{
public:

    using ExtentsType = std::experimental::extents<std::size_t, std::dynamic_extent, TDim>;

    using MatrixViewType = std::experimental::mdspan<double, ExtentsType>;

    static void ApplyGradientOperator(
        const std::array<int, TDim>& rBoxDivisions,
        const std::array<double, TDim>& rCellSize,
        const std::vector<bool>& rActiveCells,
        const src_accessor_t & rX,
        MatrixViewType& rOutput);

    static void ApplyDivergenceOperator(
        const std::array<int, TDim>& rBoxDivisions,
        const std::array<double, TDim>& rCellSize,
        const std::vector<bool>& rActiveCells,
        const MatrixViewType& rX,
        dst_accessor_t & rOutput);

};