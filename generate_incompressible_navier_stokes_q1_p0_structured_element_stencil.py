import math
import sympy
import importlib
import numpy as np
import KratosMultiphysics
import KratosMultiphysics.sympy_fe_utilities as KratosSympy

import mesher
import incompressible_navier_stokes_q1_p0_structured_element_2d as element

# Define cell data
a = sympy.Symbol('a', positive = True)
b = sympy.Symbol('b', positive = True)

# Mesh data
box_size = [3*a,3*b,None]
box_divisions = [3,3,None]
cell_size = [i/j if i is not None else 0.0 for i, j in zip(box_size, box_divisions)]
if box_size[2] == None:
    dim = 2
    box_divisions[2] = 1
else:
    dim = 3

# Create mesh
nodes, cells = mesher.CreateStructuredMesh(box_divisions, box_size, dim)
num_nodes = nodes.shape[0]
num_cells = cells.shape[0]
cell_gradient_operator = element.GetCellGradientOperator(a, b, None)
gradient_operator = np.zeros((num_nodes * dim, num_cells))
for i_cell in range(cells.shape[0]):
    cell = cells[i_cell]
    i_node = 0
    for node in cell:
        for d in range(dim):
            gradient_operator[node*dim + d, i_cell] = cell_gradient_operator[i_node*dim + d]
        i_node += 1

print(gradient_operator)
