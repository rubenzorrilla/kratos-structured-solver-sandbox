import os
import numpy as np

import KratosMultiphysics
from KratosMultiphysics.gid_output_process import GiDOutputProcess
from KratosMultiphysics.vtu_output_process import VtuOutputProcess

def parse_arrays(file_path, delimiter = " ", dtype = float):
    arrays = []
    with open(file_path, 'r') as file:
        for line in file:
            values = line.strip().split(delimiter)
            values = [dtype(val) for val in values]
            arrays.append(values)
    return arrays

def parse_arrays_time(file_path, delimiter = " "):
    arrays = []
    with open(file_path, 'r') as file:
        time = float(file.readline())
        for line in file.readlines():
            values = line.strip().split(delimiter)
            values = [float(val) for val in values]
            arrays.append(values)
    return time, np.array(arrays)

# Parse coordinates
coordinates = parse_arrays("coordinates.txt", dtype=float)

# Parse connectivities
connectivities = parse_arrays("connectivities.txt", dtype=int)

# Create auxiliary Kratos output mesh
model = KratosMultiphysics.Model()
output_model_part = model.CreateModelPart("OutputModelPart")
fake_properties = output_model_part.CreateNewProperties(0)

aux_id = 0
for node in coordinates:
    aux_id += 1
    output_model_part.CreateNewNode(aux_id, node[0], node[1], node[2])

dim = 2 if len(connectivities[0]) == 4 else 3
elem_name = "Element2D4N" if dim == 2 else "Element3D8N"
aux_id = 0
for cell in connectivities:
    aux_id += 1
    output_model_part.CreateNewElement(elem_name, aux_id, [id + 1 for id in cell], fake_properties)

output_file = "gid_output/output_model_part"
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
                "nodal_nonhistorical_results": ["VELOCITY"],
                "nodal_flags_results": [],
                "gauss_point_results": [],
                "additional_list_files": []
            }
        }
    """))
gid_output.ExecuteInitialize()
gid_output.ExecuteBeforeSolutionLoop()

vtu_output = VtuOutputProcess(
    model,
    KratosMultiphysics.Parameters("""{
        "model_part_name"                   : "OutputModelPart",
        "file_format"                       : "binary",
        "output_precision"                  : 7,
        "output_control_type"               : "step",
        "output_interval"                   : 1.0,
        "output_sub_model_parts"            : false,
        "output_path"                       : "vtu_output",
        "save_output_files_in_folder"       : true,
        "write_deformed_configuration"      : false,
        "nodal_solution_step_data_variables": [],
        "nodal_data_value_variables"        : ["VELOCITY"],
        "nodal_flags"                       : [],
        "element_data_value_variables"      : ["PRESSURE"],
        "element_flags"                     : [],
        "condition_data_value_variables"    : [],
        "condition_flags"                   : []
    }"""))
vtu_output.ExecuteInitialize()
vtu_output.ExecuteBeforeSolutionLoop()

step_vect = []
for filename in os.listdir(os.getcwd()):
    if filename.startswith("p_"):
        step = int(filename.strip(".txt").split("_")[1])
        step_vect.append(step)
step_vect.sort()

# Time loop
for step in step_vect:

    # Parse current step results
    time_p, p_array = parse_arrays_time(f"p_{step}.txt")
    time_v, v_array = parse_arrays_time(f"v_{step}.txt", delimiter = ", ")
    if (abs(time_p - time_v) > 1.0e-12):
        raise Exception("Velocity and pressure data are from different time step.")

    # Output results
    output_model_part.CloneTimeStep(time_p)
    output_model_part.ProcessInfo[KratosMultiphysics.STEP] = step
    output_model_part.ProcessInfo[KratosMultiphysics.TIME] = time_p

    # KratosMultiphysics.VariableUtils().SetValuesVector(output_model_part.Nodes, KratosMultiphysics.DISTANCE, list(distance))
    # KratosMultiphysics.VariableUtils().SetValuesVector(output_model_part.Nodes, KratosMultiphysics.DISPLACEMENT_X, list(distance_vects[:,0]))
    # KratosMultiphysics.VariableUtils().SetValuesVector(output_model_part.Nodes, KratosMultiphysics.DISPLACEMENT_Y, list(distance_vects[:,1]))
    KratosMultiphysics.VariableUtils().SetValuesVector(output_model_part.Nodes, KratosMultiphysics.VELOCITY_X, list(v_array[:,0]))
    KratosMultiphysics.VariableUtils().SetValuesVector(output_model_part.Nodes, KratosMultiphysics.VELOCITY_Y, list(v_array[:,1]))
    # KratosMultiphysics.VariableUtils().SetValuesVector(output_model_part.Nodes, KratosMultiphysics.ACCELERATION_X, list(acc[:,0]))
    # KratosMultiphysics.VariableUtils().SetValuesVector(output_model_part.Nodes, KratosMultiphysics.ACCELERATION_Y, list(acc[:,1]))
    # KratosMultiphysics.VariableUtils().SetValuesVector(output_model_part.Nodes, KratosMultiphysics.VOLUME_ACCELERATION_X, list(f[:,0]))
    # KratosMultiphysics.VariableUtils().SetValuesVector(output_model_part.Nodes, KratosMultiphysics.VOLUME_ACCELERATION_Y, list(f[:,1]))
    # KratosMultiphysics.VariableUtils().SetValuesVector(output_model_part.Nodes, KratosMultiphysics.NODAL_AREA, list(lumped_mass_vector[:,0]))
    # KratosMultiphysics.VariableUtils().SetValuesVector(output_model_part.Nodes, KratosMultiphysics.YOUNG_MODULUS, [float(i) for i in surrogate_nodes])
    if dim == 3:
        KratosMultiphysics.VariableUtils().SetValuesVector(output_model_part.Nodes, KratosMultiphysics.VELOCITY_Z, list(v_array[:,2]))
        # KratosMultiphysics.VariableUtils().SetValuesVector(output_model_part.Nodes, KratosMultiphysics.ACCELERATION_Z, list(acc[:,2]))
        # KratosMultiphysics.VariableUtils().SetValuesVector(output_model_part.Nodes, KratosMultiphysics.VOLUME_ACCELERATION_Z, list(f[:,2]))

    gid_output.ExecuteInitializeSolutionStep()
    gid_output.PrintOutput()
    gid_output.ExecuteFinalizeSolutionStep()

    vtu_output.ExecuteInitializeSolutionStep()
    vtu_output.PrintOutput()
    vtu_output.ExecuteFinalizeSolutionStep()

# Finalize results
gid_output.ExecuteFinalize()
vtu_output.ExecuteFinalize()
