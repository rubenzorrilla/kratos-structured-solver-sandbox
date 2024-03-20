#include <array>
#include <utility>
#include <vector>
#include <Eigen/Dense>

#pragma once

class CellUtilities
{
public:

    static unsigned int GetCellGlobalId(
        const unsigned int I,
        const unsigned int J,
        const std::array<int, 2>& rBoxDivisions)
    {
        return I + J * rBoxDivisions[0];
    }

    static unsigned int GetCellGlobalId(
        const unsigned int I,
        const unsigned int J,
        const unsigned int K,
        const std::array<int, 3>& rBoxDivisions)
    {
        return I + J * rBoxDivisions[0] + K * rBoxDivisions[1];
    }

    static unsigned int GetNodeGlobalId(
        const unsigned int I,
        const unsigned int J,
        const std::array<int, 2>& rBoxDivisions)
    {
        return I + J * (rBoxDivisions[0] + 1);
    }

    static unsigned int GetNodeGlobalId(
        const unsigned int I,
        const unsigned int J,
        const unsigned int K,
        const std::array<int, 3>& rBoxDivisions)
    {
        return I + J * (rBoxDivisions[0] + 1) + K * (rBoxDivisions[1] + 1);
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
};