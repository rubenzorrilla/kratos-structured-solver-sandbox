#include <array>
#include <utility>
#include <vector>

#pragma once

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

    static void GetCellNodesGlobalIds(
        const unsigned int I,
        const unsigned int J,
        const std::array<int, 2>& rBoxDivisions,
        std::array<int, 4>& rCellIds);

    static void GetCellNodesGlobalIds(
        const unsigned int I,
        const unsigned int J,
        const unsigned int K,
        const std::array<int, 3>& rBoxDivisions,
        std::array<int, 8>& rCellIds);

    static double GetCellDomainSize(const std::array<double, 2>& rCellSize)
    {
        return rCellSize[0]*rCellSize[1];
    }

    static double GetCellDomainSize(const std::array<double, 3>& rCellSize)
    {
        return rCellSize[0]*rCellSize[1]*rCellSize[2];
    }
};