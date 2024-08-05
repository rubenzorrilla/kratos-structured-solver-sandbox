#pragma once

#include <array>
#include <utility>
#include <vector>

// Intel sycl
#include <CL/sycl.hpp> 

#include "include/experimental/mdspan"

template<int TDim>
class MeshUtilities
{
public:

    static constexpr std::size_t cell_nodes = TDim == 2 ? 4 : 8;

    static constexpr std::size_t cell_dofs = cell_nodes * TDim;

    using ExtentsType = std::experimental::extents<std::size_t, std::dynamic_extent, TDim>;

    using MatrixViewType = std::experimental::mdspan<double, ExtentsType>;

    using FixityMatrixViewType = std::experimental::mdspan<bool, ExtentsType>;

    SYCL_EXTERNAL static std::pair<unsigned int, unsigned int> CalculateMeshData(const std::array<int, TDim>& rBoxDivisions) {
        if constexpr (TDim == 2) {
            const unsigned int num_cells = rBoxDivisions[0] * rBoxDivisions[1];
            const unsigned int num_nodes = (rBoxDivisions[0] + 1) * (rBoxDivisions[1] + 1);
            return std::make_pair(num_nodes, num_cells);
        } else {
            const unsigned int num_cells = rBoxDivisions[0] * rBoxDivisions[1] * rBoxDivisions[2];
            const unsigned int num_nodes = (rBoxDivisions[0] + 1) * (rBoxDivisions[1] + 1) * (rBoxDivisions[2] + 1);
            return std::make_pair(num_nodes, num_cells);
        }
    }

    SYCL_EXTERNAL static std::array<double, TDim> CalculateCellSize(
        const std::array<double, TDim>& rBoxSize,
        const std::array<int, TDim>& rBoxDivisions);

    static void CalculateNodalCoordinates(
        const std::array<double, TDim>& rBoxSize,
        const std::array<int, TDim>& rBoxDivisions,
        MatrixViewType& rNodalCoords);

    static void CalculateLumpedMassVector(
        const double MassFactor,
        const std::array<int, TDim>& rBoxDivisions,
        MatrixViewType& rLumpedMassVector);

    static void CalculateLumpedMassVector(
        const double MassFactor,
        const std::array<int, TDim>& rBoxDivisions,
        const std::vector<bool>& rActiveCells,
        MatrixViewType& rLumpedMassVector);

    static std::tuple<bool, unsigned int> FindFirstFreeCellId(
        const std::array<int, TDim> &rBoxDivisions,
        const FixityMatrixViewType& rFixity,
        const std::vector<bool>& rActiveCells);

    static void OutputVector(
        const std::vector<double>& rVector,
        const std::string Filename,
        const std::string OutputPath = "");
};