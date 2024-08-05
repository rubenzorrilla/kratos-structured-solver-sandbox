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
    // throw std::logic_error("'GetNeighbourCellsGlobalIds' is not implemented for 3D yet.");
}

void CellUtilities::GetCellMidpointCoordinates(
    const unsigned int I,
    const unsigned int J,
    const std::array<double, 2>& rCellSize,
    std::array<double, 2>& rCellMidpointCoordinates)
{
    rCellMidpointCoordinates[0] = (J + 0.5) * rCellSize[0];
    rCellMidpointCoordinates[1] = (I + 0.5) * rCellSize[1];
}

void CellUtilities::GetCellMidpointCoordinates(
    const unsigned int I,
    const unsigned int J,
    const unsigned int K,
    const std::array<double, 3>& rCellSize,
    std::array<double, 3>& rCellMidpointCoordinates)
{
    rCellMidpointCoordinates[0] = (J + 0.5) * rCellSize[0];
    rCellMidpointCoordinates[1] = (I + 0.5) * rCellSize[1];
    rCellMidpointCoordinates[2] = (K + 0.5) * rCellSize[2];
}
