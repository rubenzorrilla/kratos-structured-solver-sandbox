import math
import numpy
import sympy
import importlib
import KratosMultiphysics
import KratosMultiphysics.sympy_fe_utilities as KratosSympy

def GetOutstring(dim):
    if dim == 2:
        outstring = "import math\n"
        outstring += "import numpy\n"
        outstring += "\n"
        outstring += "def CalculateRightHandSide(a, b, c, x0, y0, z0, mu, rho, v, p, f, acc, v_conv, stab_c1 = 4.0, stab_c2 = 2.0):\n"
        outstring += f"    RHS = numpy.empty(8)\n"
        outstring += "#substitute_rhs_2d"
        outstring += "\n    return RHS"
    elif dim == 3:
        outstring = "import math\n"
        outstring += "import numpy\n"
        outstring += "\n"
        outstring += "def CalculateRightHandSide(a, b, c, x0, y0, z0, mu, rho, v, p, f, acc, v_conv, stab_c1 = 4.0, stab_c2 = 2.0):\n"
        outstring += f"    RHS = numpy.empty(16)\n"
        outstring += "#substitute_rhs_3d"
        outstring += "\n    return RHS"
    else:
        raise NotImplementedError

    return outstring

def ImportKinematicsModule(dim):
    if dim == 2:
        return importlib.import_module("quadrilateral_2d_4n_kinematics")
    elif dim == 3:
        return importlib.import_module("hexahedron_3d_8n_kinematics")
    else:
        raise NotImplementedError

dim = 2
do_simplify = False
kinematics_module = ImportKinematicsModule(dim)

# Cell data
a = sympy.Symbol('a', positive = True)
b = sympy.Symbol('b', positive = True)
c = None
x0 = sympy.Symbol('x0', positive = True)
y0 = sympy.Symbol('y0', positive = True)
z0 = None

# Material data
mu = sympy.Symbol('mu', positive=True)
rho = sympy.Symbol('rho', positive=True)

# Stabilization parameters
stab_c1 = sympy.Symbol('stab_c1', positive=True)
stab_c2 = sympy.Symbol('stab_c2', positive=True)

# Unknown fields
num_nodes = 4 if dim == 2 else 8
p = sympy.Symbol('p') # Pressure value
f = KratosSympy.DefineMatrix('f', num_nodes, dim) # Body force
v = KratosSympy.DefineMatrix('v', num_nodes, dim) # Nodal velocities
acc = KratosSympy.DefineMatrix('acc', num_nodes, dim) # Nodal acceleration (previous step)
v_conv = KratosSympy.DefineMatrix('v_conv', num_nodes, dim) # Linearised convective velocity

# Test functions
w = KratosSympy.DefineMatrix('w', num_nodes, dim) # Test function nodal values

# Get quadrature and parametric nodal coordinates
integration_order = 2
quadrature = kinematics_module.GetGaussQuadrature(integration_order)
nodal_coords = kinematics_module.SetNodalCoordinates(x0, y0, z0, a, b, c)

# Loop the Gauss points (note that this results in the complete elemental RHS contribution)
RHS = sympy.Matrix(num_nodes*dim, 1, lambda i, j : 0.0)
for g in range(len(quadrature)):
    # Get current Gauss point data
    gauss_coords = quadrature[g][0]
    gauss_weight = quadrature[g][1]

    # Calculate current Gauss point kinematics
    jacobian = kinematics_module.CalculateJacobian(gauss_coords, nodal_coords)
    weight = gauss_weight*jacobian.det()
    N = kinematics_module.ShapeFunctionsLocalValues(gauss_coords)
    DN_DX = kinematics_module.ShapeFunctionsGradients(gauss_coords, jacobian)
    DDN_DDX = kinematics_module.ShapeFunctionsSecondGradients(nodal_coords, gauss_coords, jacobian)

    # Define Gauss point interpolations
    f_g = N * f
    w_g = N * w
    acc_g = N * acc
    v_conv_g = N *v_conv

    grad_v_g = DN_DX.transpose() * v
    grad_w_g = DN_DX.transpose() * w
    grad_v_conv_g = DN_DX.transpose() * v_conv

    div_w_g = sum([grad_w_g[d,d] for d in range(dim)])
    div_v_conv_g = sum([grad_v_conv_g[d,d] for d in range(dim)])

    # Calculate current Gauss point stabilization constant
    # Note that the element size (h) is computed at each Gauss point
    h = sympy.sqrt(weight) if dim == 2 else sympy.cbrt(weight)
    v_conv_g_norm = sympy.sqrt(sum(v**2 for v in v_conv_g))
    tau = 1.0/(stab_c1*mu/h**2 + stab_c2*rho*v_conv_g_norm/h)

    # Calculate Gauss point functional terms
    pres_term = div_w_g*p
    forcing_term = (rho*w_g*f_g.transpose())[0,0]
    conv_term = (rho*w_g*(v_conv_g @ grad_v_g).transpose())[0,0]
    visc_term = mu*KratosSympy.DoubleContraction(grad_w_g, grad_v_g)

    # Define velocity subscale
    v_subs_forcing_term = rho*f_g.transpose()
    v_subs_inertial_term = rho*acc_g.transpose()
    v_subs_conv_term = rho*(v_conv_g @ grad_v_g).transpose()
    v_subs_visc_term = sympy.Matrix(2, 1, lambda i, j : 0.0)
    for i in range(num_nodes):
        DDN_DDX_i = DDN_DDX[i,:,:]
        v_subs_visc_term += DDN_DDX_i@(v[i,:]).transpose()
    v_subs = tau * (v_subs_forcing_term - v_subs_inertial_term - v_subs_conv_term + v_subs_visc_term)

    # Calculate Gauss point stabilization functional terms
    stab_conv_term_1 = (rho * div_v_conv_g * w_g * v_subs)[0,0]
    stab_conv_term_2 = (rho * v_conv_g @ grad_w_g * v_subs)[0,0]

    # Add and differentiate the functional
    phi = forcing_term - conv_term - visc_term + pres_term
    phi += stab_conv_term_1 + stab_conv_term_2

    for i in range(num_nodes):
        for d in range(dim):
            rhs_i_d = sympy.diff(phi, w[i,d])
            if do_simplify:
                rhs_i_d = sympy.simplify(rhs_i_d)
            RHS[i*dim + d] += gauss_weight * rhs_i_d

RHS_output = KratosSympy.OutputVector_CollectingFactors(RHS, "RHS", "python", indentation_level=1, replace_indices=True, assignment_op=" = ")
outstring = GetOutstring(dim)
outstring = outstring.replace(f"#substitute_rhs_{dim}d", RHS_output)

out = open(f"incompressible_navier_stokes_q1_p0_structured_element_{dim}d.py",'w')
out.write(outstring)
out.close()

