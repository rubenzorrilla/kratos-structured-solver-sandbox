import mesher
import numpy as np
import incompressible_navier_stokes_q1_p0_structured_element_2d as element

def CalculateDeltaTime(cell_size, velocities, target_cfl):
    max_v = 0.0
    for v in velocities:
        norm_v = np.linalg.norm(v)
        if norm_v > max_v:
            max_v = norm_v

    dt_x = cell_size[0] * target_cfl / max_v
    dt_y = cell_size[1] * target_cfl / max_v
    dt_z = cell_size[2] * target_cfl / max_v if cell_size[2] > 1.0e-12 else 1.0e8

    return min(dt_x, dt_y, dt_z)

def GetButcherTableau():
    A = np.zeros((4,4))
    A[1,0] = 0.5
    A[2,1] = 0.5
    A[3,2] = 1.0
    B = np.array([0.0, 0.5, 0.5, 1.0])
    C = np.array([1.0/6.0, 1.0/3.0, 1.0/3.0, 1.0/6.0])
    return A, B, C

# Problem data
dt = 0.1
end_time = 0.05
init_time = 0.0

# Material data
mu = 1.0
rho = 1.0

# Mesh data
box_size = [1.0,1.0,None]
box_divisions = [10,1,None]
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

# TODO: Create auxiliary Kratos output mesh

# Create mesh dataset
p = np.zeros((num_cells, 1))
f = np.zeros((num_nodes, 3))
v = np.zeros((num_nodes, 3))
v_n = np.zeros((num_nodes, 3))
acc = np.zeros((num_nodes, 3))

cell_domain_size = np.prod(cell_size[:dim])
mass_factor = (rho*cell_domain_size) / (4.0 if dim == 2 else 8.0)
lumped_mass_vector = np.zeros((num_nodes * dim, 1))
for cell in cells:
    for node in cell:
        for d in range(dim):
            lumped_mass_vector[node*dim + d] += mass_factor

# Set velocity fixity vector (0: free ; 1: fixed)
fixity = np.zeros((num_nodes*dim, 1), dtype=int)

tol = 1.0e-6
for i_node in range(num_nodes):
    node = nodes[i_node]
    if node[0] < tol:
        fixity[i_node*dim] = 1
    if ((node[1] < tol) or (node[1] > (1.0-tol))):
        fixity[i_node*dim + 1] = 1

# Set initial conditions
for i_node in range(num_nodes):
    node = nodes[i_node]
    if node[0] < tol:
        v[i_node, :] = [1.0,0.0,0.0]
        v_n[i_node, :] = [1.0,0.0,0.0]

print(v)
print(v_n)

# Time loop
current_step = 1
current_time = init_time
rk_A, rk_B, rk_C = GetButcherTableau()
while current_time < end_time:
    dt = CalculateDeltaTime(cell_size, v_n, 0.5)
    print(f"### Step {current_step} - time {current_time} - dt {dt} ###\n")

    #TODO: IMPLEMENT FIXITY!!!
    # Solve intermediate velocity with RK scheme
    rk_step_time = current_time
    rk_num_steps = rk_C.shape[0]
    rk_res = np.zeros((num_nodes*dim, rk_num_steps))
    for rk_step in range(rk_num_steps):
        # Calculate input values for current step residual calculation
        rk_step_time += rk_C[rk_step]*dt

        rk_v = np.empty((num_nodes*dim, 1))
        for a_ij in rk_A[rk_step, :rk_step]:
            for i in range(num_nodes):
                for d in range(dim):
                    rk_v[i*dim + d] = a_ij * rk_res[i * dim + d, rk_step]
        for i in range(num_nodes):
            for d in range(dim):
                aux_i = i * dim + d
                rk_v[aux_i] *= dt / lumped_mass_vector[aux_i, 0]
                rk_v[aux_i] += v_n[i,d]

        # Calculate current step residual
        for i_cell in range(num_cells):
            # Get current cell data
            cell_p = p[i_cell]
            cell_v = np.empty((4 if dim == 2 else 8, dim))
            cell_f = np.empty((4 if dim == 2 else 8, dim))
            cell_acc = np.empty((4 if dim == 2 else 8, dim))
            aux_i = 0
            for i_node in cells[i_cell]:
                for d in range(dim):
                    cell_f[aux_i, d] = f[i_node, d]
                    cell_acc[aux_i, d] = acc[i_node, d]
                    cell_v[aux_i, d] = rk_v[i_node * dim + d, 0]
                aux_i += 1

            # Calculate current cell residual
            cell_res = element.CalculateRightHandSide(cell_size[0], cell_size[1], cell_size[2], mu, rho, cell_v, cell_p, cell_f, cell_acc, cell_v)

            # Assemble current cell residual
            aux_i = 0
            for i_node in cells[i_cell]:
                for d in range(dim):
                    rk_res[i_node * dim + d, rk_step] += cell_res[aux_i * dim + d]
                aux_i += 1

    for i in range(num_nodes):
        for d in range(dim):
            aux_i = i * dim + d
            for rk_step in range(rk_num_steps):
                v[i, d] += rk_B[rk_step] * rk_res[aux_i, rk_step]
            v[i, d] *= dt * lumped_mass_vector[aux_i, 0]
            v[i, d] += v_n[i, d]

    # Solve pressure update

    # Correct velocity

    # Output results
    print(v)
    print(v_n)

    # Update variables for next time step
    v_n = v
    current_step += 1
    current_time += dt




