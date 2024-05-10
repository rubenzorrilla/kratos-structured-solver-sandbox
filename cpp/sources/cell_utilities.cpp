#include <iostream>

#include "cell_utilities.hpp"

void CellUtilities::GetNeighbourCellsGlobalIds(
    const unsigned int I,
    const unsigned int J,
    const std::array<int, 2>& rBoxDivisions,
    std::array<int, 8>& rCellNeighboursIds)
{
    rCellNeighboursIds[0] = I > 0 && J > 0 ? GetCellGlobalId(I - 1, J - 1, rBoxDivisions) : - 1;
    rCellNeighboursIds[1] = I > 0 ? GetCellGlobalId(I - 1, J, rBoxDivisions) : - 1;
    rCellNeighboursIds[2] = I > 0 && J + 1 < rBoxDivisions[0] ? GetCellGlobalId(I - 1, J + 1, rBoxDivisions) : - 1;
    rCellNeighboursIds[3] = J > 0 ? GetCellGlobalId(I, J - 1, rBoxDivisions) : - 1;
    rCellNeighboursIds[4] = J + 1 < rBoxDivisions[0] ? GetCellGlobalId(I, J + 1, rBoxDivisions) : - 1;
    rCellNeighboursIds[5] = I + 1 < rBoxDivisions[1] && J > 0 ? GetCellGlobalId(I + 1, J - 1, rBoxDivisions) : - 1;
    rCellNeighboursIds[6] = I + 1 < rBoxDivisions[1] ? GetCellGlobalId(I + 1, J, rBoxDivisions) : - 1;
    rCellNeighboursIds[7] = I + 1 < rBoxDivisions[1] && J + 1 < rBoxDivisions[0] ? GetCellGlobalId(I + 1, J + 1, rBoxDivisions) : - 1;
}

void CellUtilities::GetNeighbourCellsGlobalIds(
    const unsigned int I,
    const unsigned int J,
    const unsigned int K,
    const std::array<int, 3>& rBoxDivisions,
    std::array<int, 24>& rCellNeighboursIds)
{
    throw std::logic_error("'GetNeighbourCellsGlobalIds' is not implemented for 3D yet.");
}

void CellUtilities::GetCellNodesGlobalIds(
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

void CellUtilities::GetCellNodesGlobalIds(
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
