#pragma once

#include <array>
#include <utility>
#include <vector>

// Intel sycl
#include <CL/sycl.hpp> 

class CellUtilities
{
public:

    static unsigned int GetCellGlobalId(
        const unsigned int I,
        const unsigned int J,
        const std::array<int, 2>& rBoxDivisions)
    {
        return J + I * rBoxDivisions[0];
    }

    static unsigned int GetCellGlobalId(
        const unsigned int I,
        const unsigned int J,
        const unsigned int K,
        const std::array<int, 3>& rBoxDivisions)
    {
        return K + I * rBoxDivisions[0] + J * rBoxDivisions[1];
    }

    static void GetNeighbourCellsGlobalIds(
        const unsigned int I,
        const unsigned int J,
        const std::array<int, 2>& rBoxDivisions,
        std::array<int, 8>& rCellNeighboursIds);

    static void GetNeighbourCellsGlobalIds(
        const unsigned int I,
        const unsigned int J,
        const unsigned int k,
        const std::array<int, 3>& rBoxDivisions,
        std::array<int, 24>& rCellNeighboursIds);

    static unsigned int GetNodeGlobalId(
        const unsigned int I,
        const unsigned int J,
        const std::array<int, 2>& rBoxDivisions)
    {
        return J + I * (rBoxDivisions[0] + 1);
    }

    static unsigned int GetNodeGlobalId(
        const unsigned int I,
        const unsigned int J,
        const unsigned int K,
        const std::array<int, 3>& rBoxDivisions)
    {
        return K + I * (rBoxDivisions[0] + 1) + J * (rBoxDivisions[1] + 1);
    }

    SYCL_EXTERNAL static void GetCellNodesGlobalIds(
        const unsigned int I,
        const unsigned int J,
        const std::array<int, 2>& rBoxDivisions,
        std::array<int, 4>& rCellIds)
    {
        rCellIds[0] = GetNodeGlobalId(I, J, rBoxDivisions);
        rCellIds[1] = GetNodeGlobalId(I, J + 1, rBoxDivisions);
        rCellIds[2] = GetNodeGlobalId(I + 1, J + 1, rBoxDivisions);
        rCellIds[3] = GetNodeGlobalId(I + 1, J, rBoxDivisions);
    }

    SYCL_EXTERNAL static void GetCellNodesGlobalIds(
        const unsigned int I,
        const unsigned int J,
        const unsigned int K,
        const std::array<int, 3>& rBoxDivisions,
        std::array<int, 8>& rCellIds)
    {
        rCellIds[0] = GetNodeGlobalId(I, J, K, rBoxDivisions);
        rCellIds[1] = GetNodeGlobalId(I, J + 1, K, rBoxDivisions);
        rCellIds[2] = GetNodeGlobalId(I + 1, J + 1, K, rBoxDivisions);
        rCellIds[3] = GetNodeGlobalId(I + 1, J, K, rBoxDivisions);
        rCellIds[4] = GetNodeGlobalId(I, J, K + 1, rBoxDivisions);
        rCellIds[5] = GetNodeGlobalId(I, J + 1, K + 1, rBoxDivisions);
        rCellIds[6] = GetNodeGlobalId(I + 1, J + 1, K + 1, rBoxDivisions);
        rCellIds[7] = GetNodeGlobalId(I + 1, J, K + 1, rBoxDivisions);
    }

    static double GetCellDomainSize(const std::array<double, 2>& rCellSize)
    {
        return rCellSize[0]*rCellSize[1];
    }

    static double GetCellDomainSize(const std::array<double, 3>& rCellSize)
    {
        return rCellSize[0]*rCellSize[1]*rCellSize[2];
    }

    static void GetCellMidpointCoordinates(
        const unsigned int I,
        const unsigned int J,
        const std::array<double, 2>& rCellSize,
        std::array<double, 2>& rCellMidpointCoordinates);

    static void GetCellMidpointCoordinates(
        const unsigned int I,
        const unsigned int J,
        const unsigned int K,
        const std::array<double, 3>& rCellSize,
        std::array<double, 3>& rCellMidpointCoordinates);
};