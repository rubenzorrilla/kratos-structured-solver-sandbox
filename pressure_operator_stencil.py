import sympy
import numpy as np

import generate_incompressible_navier_stokes_q1_p0_structured_element as generator

def CalculatePressureOperatorStencil(dim, a, b , c):

    # Define cell data
    a_aux = sympy.Symbol('a', positive = True)
    b_aux = sympy.Symbol('b', positive = True)
    c_aux = sympy.Symbol('c', positive = True)

    # Auxiliary stencil mesh data
    box_size = [3*a_aux, 3*b_aux, 3*c_aux if dim == 3 else None]
    box_divisions = [3, 3, 1 if dim == 3 else None]

    # Retrieve mesh parameters
    nx = box_divisions[0] + 1
    ny = box_divisions[1] + 1
    nz = box_divisions[2] + 1 if dim == 3 else None
    Lx = box_size[0]
    Ly = box_size[1]
    Lz = box_size[2] if dim == 3 else 0.0

    # Generate nodes
    x = [i*Lx for i in range(nx)]
    y = [i*Ly for i in range(ny)]
    if dim == 2:
        nodes = np.array([[xx, yy] for yy in y for xx in x])
    elif dim == 3:
        z = [i*Lz for i in range(nz)]
        nodes = np.array([[xx, yy, zz] for zz in z for yy in y for xx in x])
    else:
        raise ValueError

    # Generate elements
    cells = []
    for i in range(nz - 1) if dim == 3 else range(ny - 1):
        for j in range(nx - 1):
            if dim == 2:
                node1 = i * nx + j
                node2 = node1 + 1
                node3 = (i + 1) * nx + j + 1
                node4 = node3 - 1
                cells.append([node1, node2, node3, node4])

            elif dim == 3:
                node1 = i * nx * ny + j * nx + 0
                node2 = node1 + 1
                node3 = (i + 1) * nx * ny + (j + 1) * nx + 1
                node4 = node3 - 1
                node5 = node1 + nx
                node6 = node2 + nx
                node7 = node3 + nx
                node8 = node4 + nx
                cells.append([node1, node2, node3, node4, node5, node6, node7, node8])

    # Create auxiliary stencil mesh
    num_nodes = len(nodes)
    num_cells = len(cells)

    # Gradient operator assembly
    integration_order = 2
    kinematics_module = generator.ImportKinematicsModule(dim)
    quadrature = kinematics_module.GetGaussQuadrature(integration_order)

    G = sympy.Matrix(num_nodes * dim, num_cells, lambda i, j : 0.0)
    lumped_mass_factor = a * b / 4.0 if dim == 2 else a * b * c / 8.0
    lumped_mass_vector = sympy.Matrix(num_nodes * dim, 1, lambda i, _ : 0.0)
    for i_cell in range(num_cells):
        # Calculate current cell nodal coordinates for the kinematics calculation
        cell = cells[i_cell]
        x0 = nodes[cell[0]][0]
        y0 = nodes[cell[0]][1]
        z0 = nodes[cell[0]][2] if dim == 3 else None
        nodal_coords = kinematics_module.SetNodalCoordinates(x0, y0, z0, a, b, c)

        # Compute current cell contribution
        n_nodes = 4 if dim == 2 else 8
        G_cell = sympy.Matrix(n_nodes*dim, 1, lambda i, j : 0.0)
        for g in range(len(quadrature)):
            # Get current Gauss point data
            gauss_coords = quadrature[g][0]
            gauss_weight = quadrature[g][1]

            # Calculate current Gauss point kinematics
            jacobian = kinematics_module.CalculateJacobian(gauss_coords, nodal_coords)
            DN_DX = kinematics_module.ShapeFunctionsGradients(gauss_coords, jacobian)

            # Add gradient (and divergence) operator contribution
            for i in range(n_nodes):
                for d in range(dim):
                    G_cell[i*dim + d, 0] += gauss_weight * jacobian.det() * DN_DX[i,d]

        # Current cell contribution assembly
        i_node = 0
        for node in cell:
            for d in range(dim):
                G[node*dim + d, i_cell] += G_cell[i_node*dim + d]
                lumped_mass_vector[node*dim + d, 0] += lumped_mass_factor
            i_node += 1

    G = sympy.simplify(G)
    lumped_mass_vector = sympy.simplify(lumped_mass_vector)

    P = sympy.Matrix(num_cells, num_cells, lambda i, j : 0.0)
    for i in range(num_cells):
        for j in range(num_cells):
            for k in range(num_nodes*dim):
                P[i,j] += G[k,i] * G[k,j] / lumped_mass_vector[k,0]

    for i in range(num_cells):
        for j in range(num_cells):
            P[i,j] = P[i,j].subs(a_aux, a)
            P[i,j] = P[i,j].subs(b_aux, b)
            P[i,j] = P[i,j].subs(c_aux, c)

    return P[4,:] # Return the row corresponding to the center cell
