import mesher
import numpy as np
import incompressible_navier_stokes_q1_p0_structured_element_2d as element

import KratosMultiphysics
from KratosMultiphysics.gid_output_process import GiDOutputProcess

def CalculateDeltaTime(rho, mu, cell_size, velocities, target_cfl, target_fourier):
    max_v = 0.0
    for v in velocities:
        norm_v = np.linalg.norm(v)
        if norm_v > max_v:
            max_v = norm_v

    tol = 1.0e-12
    if max_v < tol:
        max_v = tol

    cfl_dt_x = cell_size[0] * target_cfl / max_v
    cfl_dt_y = cell_size[1] * target_cfl / max_v
    cfl_dt_z = cell_size[2] * target_cfl / max_v if cell_size[2] > 1.0e-12 else 1.0e8
    cfl_dt = min(cfl_dt_x, cfl_dt_y, cfl_dt_z)

    fourier_dt_x = rho * cell_size[0]**2 * target_fourier / mu
    fourier_dt_y = rho * cell_size[1]**2 * target_fourier / mu
    fourier_dt_z = rho * cell_size[2]**2 * target_fourier / mu if cell_size[2] > 1.0e-12 else 1.0e8
    fourier_dt = min(fourier_dt_x, fourier_dt_y, fourier_dt_z)

    return min(cfl_dt, fourier_dt)

def GetButcherTableau():
    A = np.zeros((4,4))
    A[1,0] = 0.5
    A[2,1] = 0.5
    A[3,2] = 1.0
    B = np.array([0.0, 0.5, 0.5, 1.0])
    C = np.array([1.0/6.0, 1.0/3.0, 1.0/3.0, 1.0/6.0])
    return A, B, C

# Problem data
end_time = 5.0e-1
init_time = 0.0

# Material data
mu = 1.81e-5
rho = 1.293e0

# Mesh data
box_size = [1.0,1.0,None]
box_divisions = [3,2,None]
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

# Create auxiliary Kratos output mesh
model = KratosMultiphysics.Model()
output_model_part = model.CreateModelPart("OutputModelPart")
fake_properties = output_model_part.CreateNewProperties(0)
aux_id = 0
for node in nodes:
    aux_id += 1
    output_model_part.CreateNewNode(aux_id, node[0], node[1], 0.0)
aux_id = 0
for cell in cells:
    aux_id += 1
    output_model_part.CreateNewElement("Element2D4N", aux_id, [cell[0]+1, cell[1]+1, cell[2]+1, cell[3]+1], fake_properties)

output_file = "output_model_part"
gid_output =  GiDOutputProcess(
    output_model_part,
    output_file,
    KratosMultiphysics.Parameters("""
        {
            "result_file_configuration": {
                "gidpost_flags": {
                    "GiDPostMode": "GiD_PostAscii",
                    "WriteDeformedMeshFlag": "WriteUndeformed",
                    "WriteConditionsFlag": "WriteConditions",
                    "MultiFileFlag": "SingleFile"
                },
                "file_label": "time",
                "output_control_type": "step",
                "output_interval": 1.0,
                "body_output": true,
                "node_output": false,
                "skin_output": false,
                "plane_output": [],
                "nodal_results": [],
                "nodal_nonhistorical_results": ["VELOCITY","ACCELERATION","VOLUME_ACCELERATION"],
                "nodal_flags_results": [],
                "gauss_point_results": [],
                "additional_list_files": []
            }
        }
    """))
gid_output.ExecuteInitialize()
gid_output.ExecuteBeforeSolutionLoop()

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

cell_gradient_operator = element.GetCellGradientOperator(cell_size[0], cell_size[1], cell_size[2])
gradient_operator = np.zeros((num_nodes * dim, num_cells))
divergence_operator = np.zeros((num_cells, num_nodes * dim))
for i_cell in range(cells.shape[0]):
    cell = cells[i_cell]
    i_node = 0
    for node in cell:
        for d in range(dim):
            gradient_operator[node*dim + d, i_cell] = cell_gradient_operator[i_node*dim + d]
            divergence_operator[i_cell, node*dim + d] = cell_gradient_operator[i_node*dim + d]
        i_node += 1

# Set initial conditions
for i_node in range(num_nodes):
    v[i_node, :] = [0.0,0.0,0.0]
    v_n[i_node, :] = [1.0,0.0,0.0]
    #y_coord = nodes[i_node][1]
    #v[i_node, :] = [4.0*y_coord*(1.0-y_coord),0.0,0.0]
    #v_n[i_node, :] = [4.0*y_coord*(1.0-y_coord),0.0,0.0]

# Set velocity fixity vector (0: free ; 1: fixed) and BCs
# Note that these overwrite the initial conditions above
fixity = np.zeros((num_nodes*dim, 1), dtype=int)

tol = 1.0e-6
for i_node in range(num_nodes):
    node = nodes[i_node]
    if node[0] < tol: # Inlet
        fixity[i_node*dim] = 1 # x-velocity
        fixity[i_node*dim + 1] = 1 # y-velocity
        v[i_node, :] = [1.0,0.0,0.0]
        v_n[i_node, :] = [1.0,0.0,0.0]
    if ((node[1] < tol) or (node[1] > (1.0-tol))): # Top and bottom walls
        fixity[i_node*dim + 1] = 1 # y-velocity

# Set forcing term
for i_node in range(num_nodes):
    f[i_node, :] = [0.0,0.0,0.0]

# Set the matrix for the pressure problem
# Note that the velocity fixity needs to be considered in the lumped mass operator in here
#TODO: To be removed as soon as we have the CG with linear operator
pressure_matrix = np.zeros((num_cells, num_cells))
for i in range(num_cells):
    for j in range(num_cells):
        for m in range(num_nodes*dim):
            if fixity[m] == 0:
                pressure_matrix[i,j] += divergence_operator[i,m] * gradient_operator[m,j] / lumped_mass_vector[m,0]
pressure_matrix_inv = np.linalg.inv(pressure_matrix)

# Time loop
current_step = 1
current_time = init_time
rk_A, rk_B, rk_C = GetButcherTableau()
while current_time < end_time:
    # Compute time increment with CFL and Fourier conditions
    # Note that we use the current step velocity to be updated as it equals the previous one at this point
    dt = CalculateDeltaTime(rho, mu, cell_size, v, 0.2, 0.2)
    print(f"### Step {current_step} - time {current_time} - dt {dt} ###\n")

    # Calculate intermediate residuals
    rk_num_steps = rk_C.shape[0]
    rk_res = np.zeros((num_nodes*dim, rk_num_steps))
    for rk_step in range(rk_num_steps):
        # Calculate current step velocity for residual calculation
        rk_theta = rk_C[rk_step]
        rk_step_time = current_time + rk_theta * dt

        rk_v = np.zeros((num_nodes*dim, 1))
        for i_step in range(rk_step):
            a_ij = rk_A[rk_step, i_step]
            for i in range(num_nodes):
                for d in range(dim):
                    aux_i = i * dim + d
                    rk_v[aux_i] += a_ij * rk_res[aux_i, i_step]

        for i in range(num_nodes):
            for d in range(dim):
                aux_i = i * dim + d
                if fixity[aux_i] == 0:
                    rk_v[aux_i] *= dt / lumped_mass_vector[aux_i, 0]
                    rk_v[aux_i] += v_n[i,d]
                else:
                    rk_v[aux_i] = rk_theta * v[i,d] + (1.0 - rk_theta) * v_n[i,d]

        # Calculate current step residual
        for i_cell in range(num_cells):
            # Get current cell data
            cell_p = p[i_cell]
            cell_v = np.empty((4 if dim == 2 else 8, dim))
            cell_f = np.empty((4 if dim == 2 else 8, dim))
            cell_acc = np.empty((4 if dim == 2 else 8, dim))
            aux_i = 0
            for id_node in cells[i_cell]:
                for d in range(dim):
                    cell_f[aux_i, d] = f[id_node, d]
                    cell_acc[aux_i, d] = acc[id_node, d]
                    cell_v[aux_i, d] = rk_v[id_node * dim + d, 0]
                aux_i += 1

            # Calculate current cell residual
            cell_res = element.CalculateRightHandSide(cell_size[0], cell_size[1], cell_size[2], mu, rho, cell_v, cell_p, cell_f, cell_acc, cell_v)

            # Assemble current cell residual
            aux_i = 0
            for id_node in cells[i_cell]:
                for d in range(dim):
                    rk_res[id_node * dim + d, rk_step] += cell_res[aux_i * dim + d]
                aux_i += 1

    # Solve Runge-Kutta step
    for i in range(num_nodes):
        for d in range(dim):
            aux_i = i * dim + d
            if fixity[aux_i] == 0:
                v[i, d] = 0.0
                for rk_step in range(rk_num_steps):
                    v[i, d] += rk_B[rk_step] * rk_res[aux_i, rk_step]
                v[i, d] *= dt / lumped_mass_vector[aux_i, 0]
                v[i, d] += v_n[i, d]

    # Solve pressure update
    delta_p_rhs = np.zeros((num_cells, 1))
    for i in range(num_cells):
        for j in range(num_nodes):
            for d in range(dim):
                delta_p_rhs[i,0] -= divergence_operator[i, j*dim + d] * v[j, d]
    delta_p_rhs *= dt
    delta_p = pressure_matrix_inv @ delta_p_rhs
    p += delta_p

    # Correct velocity
    for i in range(num_nodes):
        for d in range(dim):
            aux_i = i * dim + d
            if fixity[aux_i] == 0:
                for j in range(num_cells):
                    v[i, d] += dt * gradient_operator[aux_i,j] * delta_p[j,0] / lumped_mass_vector[aux_i,0]

    print(f"v: ", v)
    print(f"p: ", p)

    # Output results
    output_model_part.CloneTimeStep(current_time)
    output_model_part.ProcessInfo[KratosMultiphysics.STEP] = current_step
    output_model_part.ProcessInfo[KratosMultiphysics.TIME] = current_time
    aux_id = 1
    for i_node in range(num_nodes):
        output_model_part.GetNode(aux_id).SetValue(KratosMultiphysics.VELOCITY, v[i_node, :])
        output_model_part.GetNode(aux_id).SetValue(KratosMultiphysics.ACCELERATION, acc[i_node, :])
        output_model_part.GetNode(aux_id).SetValue(KratosMultiphysics.VOLUME_ACCELERATION, f[i_node, :])
        aux_id += 1

    gid_output.ExecuteInitializeSolutionStep()
    gid_output.PrintOutput()
    gid_output.ExecuteFinalizeSolutionStep()

    # Update variables for next time step
    acc = (v - v_n) / dt
    v_n = v.copy()
    current_step += 1
    current_time += dt

    if current_step > 3:
        break

# Finalize results
gid_output.ExecuteFinalize()



