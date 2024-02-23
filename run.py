import mesher
import numpy as np

# Problem data
dt = 0.1
end_time = 1.0
init_time = 0.0

# Material data
mu = 0.001
rho = 1000.0

# Mesh data
box_size = [5.0,1.0,None]
box_divisions = [5,5,None]
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

print("### MESH DATA ###")
print(f"num_nodes: {num_nodes}")
print(f"num_cells: {num_cells}")
print(f"cell_size: {cell_size}\n")

# Create mesh dataset
p = np.zeros((num_cells, 1))
f = np.zeros((num_nodes, 3))
v = np.zeros((num_nodes, 3))
v_n = np.zeros((num_nodes, 3))
acc = np.zeros((num_nodes, 3))

# Time loop
current_step = 1
current_time = init_time + dt
while current_time < end_time:
    print(f"### Step {current_step} - time {current_time} ###\n")

    current_step += 1
    current_time += dt




