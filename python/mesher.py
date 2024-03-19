import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def CreateStructuredMesh(divisions, box_size, dim):
    """
    Create a structured mesh of quadrilaterals or hexahedra elements.

    Parameters:
    - divisions: number of divisions
    - box_size: length of the domain
    - dim: problem dimension

    Returns:
    - nodes: 2D or 3D array containing node coordinates
    - elements: 2D or 3D array containing element connectivity
    """

    # Retrieve mesh parameters
    nx = divisions[0] + 1
    ny = divisions[1] + 1
    nz = divisions[2] + 1 if dim == 3 else None
    Lx = box_size[0]
    Ly = box_size[1]
    Lz = box_size[2] if dim == 3 else 0.0

    # Generate nodes
    x = np.linspace(0, Lx, nx)
    y = np.linspace(0, Ly, ny)

    if dim == 2:
        nodes = np.array([[xx, yy] for yy in y for xx in x])
    elif dim == 3:
        z = np.linspace(0, Lz, nz)
        nodes = np.array([[xx, yy, zz] for zz in z for yy in y for xx in x])
    else:
        raise ValueError

    # Generate elements
    elements = []
    for i in range(nz - 1) if dim == 3 else range(ny - 1):
        for j in range(nx - 1):
            if dim == 2:
                node1 = i * nx + j
                node2 = node1 + 1
                node3 = (i + 1) * nx + j + 1
                node4 = node3 - 1
                elements.append([node1, node2, node3, node4])

            elif dim == 3:
                node1 = i * nx * ny + j * nx + 0
                node2 = node1 + 1
                node3 = (i + 1) * nx * ny + (j + 1) * nx + 1
                node4 = node3 - 1
                node5 = node1 + nx
                node6 = node2 + nx
                node7 = node3 + nx
                node8 = node4 + nx
                elements.append([node1, node2, node3, node4, node5, node6, node7, node8])

    return nodes, np.array(elements)

def PlotMesh(nodes, elements, dim):
    """
    Plot the structured mesh.

    Parameters:
    - nodes: 2D or 3D array containing node coordinates
    - elements: 2D or 3D array containing element connectivity
    - dim: problem dimension
    """

    if dim == 2:
        fig, ax = plt.subplots()
        ax.set_aspect('equal', 'box')

        # Plot nodes
        ax.plot(nodes[:, 0], nodes[:, 1], 'o', color='black')

        # Plot elements
        for element in elements:
            polygon = Polygon(nodes[element], edgecolor='black', fill=None)
            ax.add_patch(polygon)

        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.title('Structured Mesh of Quadrilaterals')
        plt.show()

    elif dim == 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_box_aspect([np.ptp(coord) for coord in nodes.T])

        # Plot nodes
        ax.scatter(nodes[:, 0], nodes[:, 1], nodes[:, 2], c='black', marker='o')

        # Plot elements
        for element in elements:
            vertices = nodes[element]
            ax.add_collection3d(Poly3DCollection([vertices], facecolors='none', edgecolors='black'))

        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_zlabel('Z-axis')
        ax.set_title('Structured Mesh of Hexahedra Elements')
        plt.show()

    else:
        raise ValueError
