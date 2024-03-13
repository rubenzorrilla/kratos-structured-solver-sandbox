import mesher
import numpy as np
import scipy
import scipy.linalg
import scipy.sparse.linalg
import incompressible_navier_stokes_q1_p0_structured_element_2d as element

import KratosMultiphysics
from KratosMultiphysics.gid_output_process import GiDOutputProcess

def CalculateDeltaTime(rho, mu, cell_size, velocities, target_cfl, target_fourier):
    max_v = np.max(np.linalg.norm(velocities, axis=1))
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
    B = np.array([1.0/6.0, 1.0/3.0, 1.0/3.0, 1.0/6.0])
    C = np.array([0.0, 0.5, 0.5, 1.0])
    return A, B, C

# def GetButcherTableau():
#     A = np.zeros((0,0))
#     B = np.array([1.0])
#     C = np.array([0.0])
#     return A, B, C

# Problem data
end_time = 1.0e1
init_time = 0.0

# Material data
mu = 1.81e-5
rho = 1.293e0

# Mesh data
box_size = [5.0,1.0,None]
box_divisions = [10,10,None]
cell_size = [i/j if i is not None else 0.0 for i, j in zip(box_size, box_divisions)]
if box_size[2] == None:
    dim = 2
else:
    dim = 3

# Create mesh
# Note that we discard the conectivities as we build these on the fly
nodes, _ = mesher.CreateStructuredMesh(box_divisions, box_size, dim)

def CalculateMeshData(box_divisions):
    if box_divisions[2] == None:
        num_cells = box_divisions[0] * box_divisions[1]
        num_nodes = (box_divisions[0] + 1) * (box_divisions[1] + 1)
    else:
        num_cells = box_divisions[0] * box_divisions[1] * box_divisions[2]
        num_nodes = (box_divisions[0] + 1) * (box_divisions[1] + 1) * (box_divisions[2] + 1)

    return num_nodes, num_cells

def GetCellGlobalId(i, j, k):
    cell_id = i + j * box_divisions[0]
    if k is not None:
        cell_id += k * box_divisions[1]

    return int(cell_id)

def GetNodeGlobalId(i, j, k):
    global_id = i + j * (box_divisions[0] + 1)
    if k is not None:
        global_id += k * (box_divisions[1] + 1)

    return int(global_id)

def GetCellNodesGlobalIds(i, j, k):
    cell_ids = np.empty(4, dtype=int) if k == None else np.empty(8, dtype=int)
    cell_ids[0] = GetNodeGlobalId(i, j, k)
    cell_ids[1] = GetNodeGlobalId(i + 1, j, k)
    cell_ids[2] = GetNodeGlobalId(i + 1, j + 1, k)
    cell_ids[3] = GetNodeGlobalId(i, j + 1, k)
    if k is not None:
        cell_ids[4] = GetNodeGlobalId(i, j, k + 1)
        cell_ids[5] = GetNodeGlobalId(i + 1, j, k + 1)
        cell_ids[6] = GetNodeGlobalId(i + 1, j + 1, k + 1)
        cell_ids[7] = GetNodeGlobalId(i, j + 1, k + 1)

    return cell_ids

num_nodes, num_cells = CalculateMeshData(box_divisions)

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

if dim ==  2:
    for i in range(box_divisions[0]):
        for j in range(box_divisions[1]):
            cell_id = GetCellGlobalId(i, j, None) + 1 # Kratos ids need to start by 1
            cell_node_ids = [node_id + 1 for node_id in GetCellNodesGlobalIds(i, j, None)] # Kratos ids need to start by 1
            output_model_part.CreateNewElement("Element2D4N", cell_id, cell_node_ids, fake_properties)
else:
    for i in range(box_divisions[0]):
        for j in range(box_divisions[1]):
            for k in range(box_divisions[2]):
                cell_id = GetCellGlobalId(i, j, k) + 1 # Kratos ids need to start by 1
                cell_node_ids = [node_id + 1 for node_id in GetCellNodesGlobalIds(i, j, k)] # Kratos ids need to start by 1
                output_model_part.CreateNewElement("Element3D8N", cell_id, cell_node_ids, fake_properties)

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
                "nodal_nonhistorical_results": ["VELOCITY","ACCELERATION","VOLUME_ACCELERATION","NODAL_AREA"],
                "nodal_flags_results": [],
                "gauss_point_results": [],
                "additional_list_files": []
            }
        }
    """))
gid_output.ExecuteInitialize()
gid_output.ExecuteBeforeSolutionLoop()

# Create mesh dataset
p = np.zeros((num_cells))
f = np.zeros((num_nodes, dim))
v = np.zeros((num_nodes, dim))
v_n = np.zeros((num_nodes, dim))
acc = np.zeros((num_nodes, dim))

# Set initial conditions
v.fill(0.0)
v_n.fill(0.0)

# Set velocity fixity vector (0: free ; 1: fixed) and BCs
# Note that these overwrite the initial conditions above
fixity = np.zeros((num_nodes,dim), dtype=int)

tol = 1.0e-6
for i_node in range(num_nodes):
    node = nodes[i_node]
    if node[0] < tol: # Inlet
        fixity[i_node, 0] = 1 # x-velocity
        fixity[i_node, 1] = 1 # y-velocity
        # v[i_node, 0] = 1.0
        # v_n[i_node, 0] = 1.0
        y_coord = nodes[i_node][1]
        v[i_node, 0] = 4.0*y_coord*(1.0-y_coord)
        v_n[i_node, 0] = 4.0*y_coord*(1.0-y_coord)
    if ((node[1] < tol) or (node[1] > (1.0-tol))): # Top and bottom walls
        fixity[i_node, 1] = 1 # y-velocity

free_dofs = (fixity == 0).nonzero()
fixed_dofs = (fixity == 1).nonzero()

# Set forcing term
f.fill(0.0)

# Calculate lumped mass vector
cell_domain_size = np.prod(cell_size[:dim])
mass_factor = (rho*cell_domain_size) / (4.0 if dim == 2 else 8.0)
lumped_mass_vector = np.zeros((num_nodes, dim))
if dim == 2:
    for i in range(box_divisions[0]):
        for j in range(box_divisions[1]):
            cell_node_ids = GetCellNodesGlobalIds(i, j, None)
            lumped_mass_vector[cell_node_ids, :] += mass_factor
else:
    for i in range(box_divisions[0]):
        for j in range(box_divisions[1]):
            for k in range(box_divisions[2]):
                cell_node_ids = GetCellNodesGlobalIds(i, j, k)
                lumped_mass_vector[cell_node_ids, :] += mass_factor

# Calculate inverse of the lumped mass vector
lumped_mass_vector_inv = 1.0 / lumped_mass_vector
lumped_mass_vector_inv_bcs = lumped_mass_vector_inv.copy()
lumped_mass_vector_inv_bcs[fixed_dofs] = 0.0

# Matrix-free version (without assembly) of the gradient operator application onto a pressure vector
def ApplyGradientOperator(x):
    sol = np.zeros((num_nodes,dim))
    cell_gradient_operator = element.GetCellGradientOperator(cell_size[0], cell_size[1], cell_size[2])

    if dim == 2:
        for i in range(box_divisions[0]):
            for j in range(box_divisions[1]):
                cell_id = GetCellGlobalId(i, j, None)
                cell_node_ids = GetCellNodesGlobalIds(i, j, None)
                i_node = 0
                for node_id in cell_node_ids:
                    sol[node_id, :] += cell_gradient_operator[i_node, :] * x[cell_id]
                    i_node += 1
    else:
        for i in range(box_divisions[0]):
            for j in range(box_divisions[1]):
                for k in range(box_divisions[2]):
                    cell_id = GetCellGlobalId(i, j, k)
                    cell_node_ids = GetCellNodesGlobalIds(i, j, k)
                    i_node = 0
                    for node_id in cell_node_ids:
                        sol[node_id, :] += cell_gradient_operator[i_node, :] * x[cell_id]
                        i_node += 1

    return sol

# Matrix-free version (without assembly) of the divergence operator application onto a velocity vector
def ApplyDivergenceOperator(x):
    sol = np.zeros(num_cells)
    cell_gradient_operator = element.GetCellGradientOperator(cell_size[0], cell_size[1], cell_size[2])

    if dim == 2:
        for i in range(box_divisions[0]):
            for j in range(box_divisions[1]):
                cell_id = GetCellGlobalId(i, j, None)
                cell_node_ids = GetCellNodesGlobalIds(i, j, None)
                sol[cell_id] = np.trace(cell_gradient_operator.transpose() @ x[cell_node_ids, :])
    else:
        for i in range(box_divisions[0]):
            for j in range(box_divisions[1]):
                for k in range(box_divisions[2]):
                    cell_id = GetCellGlobalId(i, j, k)
                    cell_node_ids = GetCellNodesGlobalIds(i, j, k)
                    sol[cell_id] = np.trace(cell_gradient_operator.transpose() @ x[cell_node_ids, :])

    return sol

# Matrix-free version of the pressure problem
def ApplyPressureOperator(x, lumped_mass_vector_inv):
    sol = ApplyGradientOperator(x)
    sol *= lumped_mass_vector_inv
    sol = ApplyDivergenceOperator(sol)
    return sol

# Create the preconditioner for the pressure CG solver
# For this we convert the periodic pressure matrix C (with no velocity BCs) to FFT
# The simplest way is to generate any vector x, compute the image vector y=C*X
# The transform of C is fft(y)./fft(x), as C is cyclic the transformed C must be diagonal
# The resulting transformed coefficientes should be real, because the operator is symmetric.
# Also it should be semidefinite positive (SPD) because the Laplacian operator is SPD
# The first coefficient is null, because the Laplacian is not PD, just SPD.
# But we can replace this null coefficient by anything different from 0.
# At most it would degrade the convergence of the PCG, but we will see that the convergence is OK.
x = np.zeros((num_cells))
c_row = int(box_divisions[0] * (box_divisions[1] / 2) + box_divisions[0] / 2) # We take the cell in the center of the domain
x[c_row] = 1.0
y = ApplyPressureOperator(x, lumped_mass_vector_inv)

fft_x = np.fft.fft(x)
fft_y = np.fft.fft(y)
fft_c = np.real(fft_y / fft_x)# Take the real part only (imaginary one is zero)
fft_c[0] = 1.0 # Remove the first coefficient as this is associated to the solution average

def apply_precond(r):
    fft_r = np.fft.fft(r)
    return np.real(np.fft.ifft(fft_r/fft_c))

precond = scipy.sparse.linalg.LinearOperator((num_cells, num_cells), matvec=apply_precond)

# Set the pressure operator
# Note that the velocity fixity needs to be considered in the lumped mass operator in here
def prod(x):
    y = ApplyPressureOperator(x, lumped_mass_vector_inv_bcs)
    return y.flat

pressure_op = scipy.sparse.linalg.LinearOperator((num_cells,num_cells), matvec=prod)

# Time loop
tot_p_iters = 0
current_step = 1
current_time = init_time
rk_A, rk_B, rk_C = GetButcherTableau()
while current_time < end_time:
    # Compute time increment with CFL and Fourier conditions
    # Note that we use the current step velocity to be updated as it equals the previous one at this point
    dt = CalculateDeltaTime(rho, mu, cell_size, v, 0.2, 0.2)
    print(f"### Step {current_step} - time {current_time} - dt {dt} ###")

    # Calculate intermediate residuals
    rk_num_steps = rk_C.shape[0]
    rk_res = [np.zeros((num_nodes,dim)) for _ in range(rk_num_steps)]
    for rk_step in range(rk_num_steps):
        # Calculate current step velocity for residual calculation
        rk_theta = rk_C[rk_step]
        rk_step_time = current_time + rk_theta * dt

        rk_v = np.zeros((num_nodes, dim))
        for i_step in range(rk_step):
            a_ij = rk_A[rk_step, i_step]
            rk_v += a_ij * rk_res[i_step]
        rk_v *= dt*lumped_mass_vector_inv
        rk_v += v_n
        rk_v[fixed_dofs] = rk_theta * v[fixed_dofs] + (1.0 - rk_theta) * v_n[fixed_dofs] # Set BC value in fixed DOFs

        # Calculate current step residual
        if dim == 2:
            for i in range(box_divisions[0]):
                for j in range(box_divisions[1]):
                    # Get current cell data
                    i_cell = GetCellGlobalId(i, j, None)
                    i_cell_nodes = GetCellNodesGlobalIds(i, j, None)
                    cell_p = p[i_cell]
                    cell_v = np.empty((4 if dim == 2 else 8, dim))
                    cell_f = np.empty((4 if dim == 2 else 8, dim))
                    cell_acc = np.empty((4 if dim == 2 else 8, dim))
                    aux_i = 0
                    for id_node in i_cell_nodes:
                        # for d in range(dim):
                        cell_f[aux_i, :] = f[id_node, :]
                        cell_acc[aux_i, :] = acc[id_node, :]
                        cell_v[aux_i, :] = rk_v[id_node, :]
                        aux_i += 1

                    # Calculate current cell residual
                    cell_res = element.CalculateRightHandSide(cell_size[0], cell_size[1], cell_size[2], mu, rho, cell_v, cell_p, cell_f, cell_acc, cell_v)

                    # Assemble current cell residual
                    aux_i = 0
                    for id_node in i_cell_nodes:
                        for d in range(dim):
                            rk_res[rk_step][id_node, d] += cell_res[aux_i * dim + d]
                        aux_i += 1
        else:
            raise NotImplementedError

    # Solve Runge-Kutta step
    v[free_dofs] = 0.0
    for rk_step in range(rk_num_steps):
        v[free_dofs] += rk_B[rk_step] * rk_res[rk_step][free_dofs]
    v[free_dofs] *= dt * lumped_mass_vector_inv[free_dofs]
    v[free_dofs] += v_n[free_dofs]
    print("Velocity prediction solved.")

    # Solve pressure update
    delta_p_rhs = ApplyDivergenceOperator(v)
    delta_p_rhs /= -dt

    p_iters = 0
    def nonlocal_iterate(arr):
        global p_iters
        p_iters += 1
    delta_p, converged = scipy.sparse.linalg.cg(pressure_op, delta_p_rhs, tol=1.0e-12, callback=nonlocal_iterate, M=precond)
    p += delta_p
    tot_p_iters += p_iters
    print(f"Pressure iterations: {p_iters}.")

    # Correct velocity
    v[free_dofs] += dt * lumped_mass_vector_inv[free_dofs] * ApplyGradientOperator(delta_p)[free_dofs]
    print("Velocity update finished.\n")

    # print(f"v: ", v)
    # print(f"p: ", p)

    # Output results
    output_model_part.CloneTimeStep(current_time)
    output_model_part.ProcessInfo[KratosMultiphysics.STEP] = current_step
    output_model_part.ProcessInfo[KratosMultiphysics.TIME] = current_time
    aux_id = 1
    for i_node in range(num_nodes):
        output_model_part.GetNode(aux_id).SetValue(KratosMultiphysics.NODAL_AREA, lumped_mass_vector[i_node,0])
        aux_id += 1

    KratosMultiphysics.VariableUtils().SetValuesVector(output_model_part.Nodes, KratosMultiphysics.VELOCITY_X, list(v[:,0]))
    KratosMultiphysics.VariableUtils().SetValuesVector(output_model_part.Nodes, KratosMultiphysics.VELOCITY_Y, list(v[:,1]))
    KratosMultiphysics.VariableUtils().SetValuesVector(output_model_part.Nodes, KratosMultiphysics.ACCELERATION_X, list(acc[:,0]))
    KratosMultiphysics.VariableUtils().SetValuesVector(output_model_part.Nodes, KratosMultiphysics.ACCELERATION_Y, list(acc[:,1]))
    KratosMultiphysics.VariableUtils().SetValuesVector(output_model_part.Nodes, KratosMultiphysics.VOLUME_ACCELERATION_X, list(f[:,0]))
    KratosMultiphysics.VariableUtils().SetValuesVector(output_model_part.Nodes, KratosMultiphysics.VOLUME_ACCELERATION_Y, list(f[:,1]))
    if dim == 3:
        KratosMultiphysics.VariableUtils().SetValuesVector(output_model_part.Nodes, KratosMultiphysics.VELOCITY_Z, list(v[:,2]))
        KratosMultiphysics.VariableUtils().SetValuesVector(output_model_part.Nodes, KratosMultiphysics.ACCELERATION_Z, list(acc[:,2]))
        KratosMultiphysics.VariableUtils().SetValuesVector(output_model_part.Nodes, KratosMultiphysics.VOLUME_ACCELERATION_Z, list(f[:,2]))

    gid_output.ExecuteInitializeSolutionStep()
    gid_output.PrintOutput()
    gid_output.ExecuteFinalizeSolutionStep()

    # Update variables for next time step
    acc = (v - v_n) / dt
    v_n = v.copy()
    current_step += 1
    current_time += dt

# Finalize results
gid_output.ExecuteFinalize()

# Print final data
print(f"TOTAL PRESSURE ITERATIONS: {tot_p_iters}")



