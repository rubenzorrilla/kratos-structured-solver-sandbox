#include <iostream>

#include "cell_utilities.hpp"

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
