import math
import numpy
import sympy
import importlib
import KratosMultiphysics
import KratosMultiphysics.sympy_fe_utilities as KratosSympy

def GetOutstringPython(dim):
    if dim == 2:
        outstring = "import math\n"
        outstring += "import numpy\n"
        outstring += "\n"
        outstring += "def CalculateRightHandSide(a, b, c, mu, rho, v, p, f, acc, v_conv):\n"
        outstring += f"    RHS = numpy.empty(8)\n"
        outstring += "#substitute_rhs_2d"
        outstring += "\n    return RHS"
        outstring += "\n\n"
        outstring += "def GetCellGradientOperator(a, b, c):\n"
        outstring += f"    G = numpy.empty(4,2)\n"
        outstring += "#substitute_G_2d"
        outstring += "\n    return G"
    elif dim == 3:
        outstring = "import math\n"
        outstring += "import numpy\n"
        outstring += "\n"
        outstring += "def CalculateRightHandSide(a, b, c, mu, rho, v, p, f, acc, v_conv):\n"
        outstring += f"    RHS = numpy.empty(24)\n"
        outstring += "#substitute_rhs_3d"
        outstring += "\n    return RHS"
        outstring += "\n\n"
        outstring += "def GetCellGradientOperator(a, b, c):\n"
        outstring += f"    G = numpy.empty(8,3)\n"
        outstring += "#substitute_G_3d"
        outstring += "\n    return G"
    else:
        raise NotImplementedError

    return outstring

def GetOutstringC():
    outstring = "#include \"incompressible_navier_stokes_q1_p0_structured_element.hpp\"\n"
    outstring += "\n"
    outstring += "void IncompressibleNavierStokesQ1P0StructuredElement::CalculateRightHandSide(\n"
    outstring += "    const double a,\n"
    outstring += "    const double b,\n"
    outstring += "    const double mu,\n"
    outstring += "    const double rho,\n"
    outstring += "    const Eigen::Array<double, 4, 2>& v,\n"
    outstring += "    const double p,\n"
    outstring += "    const Eigen::Array<double, 4, 2>& f,\n"
    outstring += "    const Eigen::Array<double, 4, 2>& acc,\n"
    outstring += "    Eigen::Array<double, 8, 1>& RHS)\n"
    outstring += "{\n"
    outstring += "\n//substitute_rhs_2d\n"
    outstring += "}\n"
    outstring += "\n"
    outstring += "void IncompressibleNavierStokesQ1P0StructuredElement::GetCellGradientOperator(\n"
    outstring += "    const double a,\n"
    outstring += "    const double b,\n"
    outstring += "    Eigen::Array<double, 4, 2>& G)\n"
    outstring += "{\n"
    outstring += "\n//substitute_G_2d\n"
    outstring += "}\n"
    outstring += "\n"
    outstring += "void IncompressibleNavierStokesQ1P0StructuredElement::CalculateRightHandSide(\n"
    outstring += "    const double a,\n"
    outstring += "    const double b,\n"
    outstring += "    const double c,\n"
    outstring += "    const double mu,\n"
    outstring += "    const double rho,\n"
    outstring += "    const Eigen::Array<double, 8, 3>& v,\n"
    outstring += "    const double p,\n"
    outstring += "    const Eigen::Array<double, 8, 3>& f,\n"
    outstring += "    const Eigen::Array<double, 8, 3>& acc,\n"
    outstring += "    Eigen::Array<double, 24, 1>& RHS)\n"
    outstring += "{\n"
    outstring += "\n//substitute_rhs_3d\n"
    outstring += "}\n"
    outstring += "\n"
    outstring += "void IncompressibleNavierStokesQ1P0StructuredElement::GetCellGradientOperator(\n"
    outstring += "    const double a,\n"
    outstring += "    const double b,\n"
    outstring += "    const double c,\n"
    outstring += "    Eigen::Array<double, 8, 3>& G)\n"
    outstring += "{\n"
    outstring += "\n//substitute_G_3d\n"
    outstring += "}\n"

    return outstring

def ImportKinematicsModule(dim):
    if dim == 2:
        return importlib.import_module("quadrilateral_2d_4n_kinematics")
    elif dim == 3:
        return importlib.import_module("hexahedron_3d_8n_kinematics")
    else:
        raise NotImplementedError

dim = 2
output_type = "c"
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
stab_c1 = 4.0
stab_c2 = 2.0

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
G = sympy.Matrix(num_nodes, dim, lambda i, j : 0.0)
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
        v_subs_visc_term += mu*DDN_DDX_i@(v[i,:]).transpose()
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
            RHS[i*dim + d] += weight * rhs_i_d

    # Add gradient (and divergence) operator contribution
    for i in range(num_nodes):
        for d in range(dim):
            G[i,d] += weight * DN_DX[i,d]

RHS_output = KratosSympy.OutputVector_CollectingFactors(RHS, "RHS", output_type, indentation_level=1, replace_indices=True, assignment_op=" = ")
G_output = KratosSympy.OutputMatrix_CollectingFactors(G, "G", output_type, indentation_level=1, replace_indices=True, assignment_op=" = ")

if output_type == "python":
    outstring = GetOutstringPython(dim)
    outstring = outstring.replace(f"#substitute_rhs_{dim}d", RHS_output)
    outstring = outstring.replace(f"#substitute_G_{dim}d", G_output)
    out = open(f"incompressible_navier_stokes_q1_p0_structured_element_{dim}d.py",'w')
else:
    outstring = GetOutstringC()
    outstring = outstring.replace(f"//substitute_rhs_{dim}d", RHS_output)
    outstring = outstring.replace(f"//substitute_G_{dim}d", G_output)
    out = open(f"incompressible_navier_stokes_q1_p0_structured_element.cpp",'w')

out.write(outstring)
out.close()

