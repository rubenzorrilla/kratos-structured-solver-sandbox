import math
import sympy
import KratosMultiphysics
import KratosMultiphysics.sympy_fe_utilities as KratosSympy

def SetNodalCoordinates(x0, y0, a, b):
    coords = sympy.Matrix(4,2, lambda i, j: 0.0)
    coords[0,0] = x0
    coords[0,1] = y0
    coords[1,0] = x0 + a
    coords[1,1] = y0
    coords[2,0] = x0 + a
    coords[2,1] = y0 + b
    coords[3,0] = x0
    coords[3,1] = y0 + b
    return coords

def GetGaussQuadrature(order):
    if order == 2:
        quadrature = [
            [(-math.sqrt(3)/3.0, -math.sqrt(3)/3.0), 1.0],
            [(-math.sqrt(3)/3.0, +math.sqrt(3)/3.0), 1.0],
            [(+math.sqrt(3)/3.0, -math.sqrt(3)/3.0), 1.0],
            [(+math.sqrt(3)/3.0, +math.sqrt(3)/3.0), 1.0]]
    else:
        raise Exception("Quadrature order {order} not implemented yet.")
    return quadrature

def CalculateJacobian(gauss_coordinates, nodal_coordinates):
    DN_mat = ShapeFunctionsLocalGradients(gauss_coordinates)
    jacobian = sympy.Matrix(2, 2, lambda i, j : 0.0)
    for i in range(4):
        for d1 in range(2):
            for d2 in range(2):
                jacobian[d1,d2] += DN_mat[i,d2] * nodal_coordinates[i,d1]
    return jacobian

def CalculateHessian(gauss_coordinates, nodal_coordinates):
    DDN_tensor = ShapeFunctionsLocalSecondGradients(gauss_coordinates)
    hessian = sympy.MutableDenseNDimArray(sympy.zeros(2*2*2),shape=(2,2,2))
    for i in range(4):
        for d1 in range(2):
            for d2 in range(2):
                for d3 in range(2):
                    hessian[d1,d2,d3] += DDN_tensor[i,d2,d3] * nodal_coordinates[i,d1]
    return hessian

def ShapeFunctionsLocalValues(gauss_coordinates):
    xi = gauss_coordinates[0]
    eta = gauss_coordinates[1]
    N = sympy.Matrix(1, 4, lambda i, j: 0.0)
    N[0] = 0.25*(1.0-xi)*(1-eta)
    N[1] = 0.25*(1.0+xi)*(1-eta)
    N[2] = 0.25*(1.0+xi)*(1+eta)
    N[3] = 0.25*(1.0-xi)*(1+eta)
    return N

def ShapeFunctionsLocalGradients(gauss_coordinates):
    # Declare symbols for local coordinates
    xi = sympy.var('xi')
    eta = sympy.var('eta')
    loc_coords = [xi, eta]

    # Declare and differentiate shape functions WRT the previous symbols
    N = sympy.Matrix(1, 4, lambda i, j: 0.0)
    N[0] = 0.25*(1.0-xi)*(1-eta)
    N[1] = 0.25*(1.0+xi)*(1-eta)
    N[2] = 0.25*(1.0+xi)*(1+eta)
    N[3] = 0.25*(1.0-xi)*(1+eta)

    DN_DE = sympy.Matrix(4, 2, lambda i, j: 0.0)
    for i in range(4):
        for j in range(2):
            DN_DE[i,j] = sympy.diff(N[i], loc_coords[j])

    # Substitute the symbols by current Gauss point coordinates
    DN_DE = DN_DE.subs(xi, gauss_coordinates[0])
    DN_DE = DN_DE.subs(eta, gauss_coordinates[1])

    return DN_DE

def ShapeFunctionsGradients(gauss_coordinates, jacobian):
    inverse_jacobian = jacobian.inv()
    DN_DE = ShapeFunctionsLocalGradients(gauss_coordinates)
    DN_DX = sympy.Matrix(4,2, lambda i, j: 0.0)
    for i in range(4):
        for d1 in range(2):
            for d2 in range(2):
                DN_DX[i,d1] += inverse_jacobian[d1,d2]*DN_DE[i,d2]
    return DN_DX

def ShapeFunctionsLocalSecondGradients(gauss_coordinates):
    # Declare symbols for local coordinates
    xi = sympy.var('xi')
    eta = sympy.var('eta')
    loc_coords = [xi, eta]

    # Declare and differentiate shape functions WRT the previous symbols
    N = sympy.Matrix(1, 4, lambda i, j: 0.0)
    N[0] = 0.25*(1.0-xi)*(1-eta)
    N[1] = 0.25*(1.0+xi)*(1-eta)
    N[2] = 0.25*(1.0+xi)*(1+eta)
    N[3] = 0.25*(1.0-xi)*(1+eta)

    DDN_DEE = sympy.MutableDenseNDimArray(sympy.zeros(4*2*2),shape=(4,2,2))
    for i in range(4):
        for j in range(2):
            Ni_Dj = sympy.diff(N[i], loc_coords[j])
            for k in range(2):
                DDN_DEE[i,j,k] = sympy.diff(Ni_Dj, loc_coords[k])

    # Substitute the symbols by current Gauss point coordinates
    for i in range(4):
        for j in range(2):
            for k in range(2):
                DDN_DEE[i,j,k] = DDN_DEE[i,j,k].subs(xi, gauss_coordinates[0])
                DDN_DEE[i,j,k] = DDN_DEE[i,j,k].subs(eta, gauss_coordinates[1])

    return DDN_DEE

def ShapeFunctionsSecondGradients(gauss_coordinates, jacobian):
    hessian = CalculateHessian(gauss_coordinates, nodal_coords)
    DN_DX = ShapeFunctionsGradients(gauss_coordinates, jacobian)
    DDN_DDE = ShapeFunctionsLocalSecondGradients(gauss_coordinates)

    LHS = sympy.Matrix(3, 3, lambda i, j: 0.0)
    LHS[0,0] = jacobian[0,0]**2
    LHS[0,1] = jacobian[1,0]**2
    LHS[0,2] = 2.0*jacobian[0,0]*jacobian[1,0]
    LHS[1,0] = jacobian[0,1]**2
    LHS[1,1] = jacobian[1,1]**2
    LHS[1,2] = 2.0*jacobian[0,1]*jacobian[1,1]
    LHS[2,0] = jacobian[0,0]*jacobian[0,1]
    LHS[2,1] = jacobian[1,0]*jacobian[1,1]
    LHS[2,2] = jacobian[0,1]*jacobian[1,0] + jacobian[0,0]*jacobian[1,1]
    inv_LHS = LHS.inv()

    DDN_DDX = sympy.MutableDenseNDimArray(sympy.zeros(4*2*2),shape=(4,2,2))
    for i in range(4):
        RHS = sympy.Matrix(3, 1, lambda i, j: 0.0)
        RHS[0] = DDN_DDE[i,0,0] - DN_DX[i,0]*hessian[0,0,0] - DN_DX[i,1]*hessian[1,0,0]
        RHS[1] = DDN_DDE[i,1,1] - DN_DX[i,0]*hessian[0,1,1] - DN_DX[i,1]*hessian[1,1,1]
        RHS[2] = DDN_DDE[i,0,1] - DN_DX[i,0]*hessian[0,0,1] - DN_DX[i,1]*hessian[1,0,1]
        DDN_DDX_i = inv_LHS*RHS
        DDN_DDX[i,0,0] = DDN_DDX_i[0]
        DDN_DDX[i,0,1] = DDN_DDX_i[2]
        DDN_DDX[i,1,0] = DDN_DDX_i[2]
        DDN_DDX[i,1,1] = DDN_DDX_i[1]

    return DDN_DDX

a = sympy.Symbol('a', positive = True)
b = sympy.Symbol('b', positive = True)
x0 = sympy.Symbol('x0', positive = True)
y0 = sympy.Symbol('y0', positive = True)
nodal_coords = SetNodalCoordinates(x0, y0, a, b)

order = 2
quadrature = GetGaussQuadrature(order)

for g in range(len(quadrature)):
    gauss_coords = quadrature[g][0]
    gauss_weight = quadrature[g][1]

    jacobian = CalculateJacobian(gauss_coords, nodal_coords)
    det_jacobian = jacobian.det()

    print(f"\ng: {g}")

    DN_DX = ShapeFunctionsGradients(gauss_coords, jacobian)
    DDN_DDX = ShapeFunctionsSecondGradients(gauss_coords, jacobian)

    sympy.pprint(DN_DX)
    sympy.pprint(DDN_DDX)










# ## Symbolic generation settings
# mode = "c"
# do_simplifications = False
# dim_to_compute = "2D"      # Spatial dimensions to compute. Options:  "2D","3D","Both"
# # dim_to_compute = "Both"      # Spatial dimensions to compute. Options:  "2D","3D","Both"
# linearisation = "Picard"     # Convective term linearisation type
# add_pressure_subscale = True # Specifies if the pressure subscale is added to the momentum equation to get the div(w)div(v) term

# # output_filename = "incompressible_navier_stokes_p2_p1_continuous.cpp"
# # template_filename = "incompressible_navier_stokes_p2_p1_continuous_cpp_template.cpp"

# info_msg = "\n"
# info_msg += "Element generator settings:\n"
# info_msg += f"\t - Dimension: {dim_to_compute}\n"
# info_msg += f"\t - Linearisation: {linearisation}\n"
# info_msg += f"\t - Pressure subscale: {add_pressure_subscale}\n"
# print(info_msg)

# if dim_to_compute == "2D":
#     dim_vector = [2]
#     v_nodes_vector = [4] # tria
#     p_nodes_vector = [1] # tria
# elif dim_to_compute == "3D":
#     dim_vector = [3]
#     v_nodes_vector = [8] # tet
#     p_nodes_vector = [1] # tet
# elif dim_to_compute == "Both":
#     dim_vector = [2, 3]
#     v_nodes_vector = [4, 8] # tria, tet
#     p_nodes_vector = [1, 1] # tria, tet

# # ## Initialize the outstring to be filled with the template .cpp file
# # print(f"Reading template file \'{template_filename}'\n")
# # templatefile = open(template_filename)
# # outstring = templatefile.read()

# for dim, v_n_nodes, p_n_nodes in zip(dim_vector, v_nodes_vector, p_nodes_vector):

#     if dim == 2:
#         strain_size = 3
#     elif dim == 3:
#         strain_size = 6

#     ## Kinematics symbols definition
#     N_v, DN_v, DDN_v = DefineShapeFunctions(v_n_nodes, dim, impose_partion_of_unity=False, shape_functions_name='N_v', first_derivatives_name='DN_v', second_derivatives_name='DDN_v')
#     N_p, DN_p = DefineShapeFunctions(p_n_nodes, dim, impose_partion_of_unity=False, shape_functions_name='N_p', first_derivatives_name='DN_p')

#     ## Unknown fields definition
#     v = DefineMatrix('r_v', v_n_nodes, dim)            # Current step velocity (v(i,j) refers to velocity of node i component j)
#     vn = DefineMatrix('r_vn', v_n_nodes, dim)          # Previous step velocity
#     vnn = DefineMatrix('r_vnn', v_n_nodes, dim)        # 2 previous step velocity
#     p = DefineVector('r_p', p_n_nodes)                 # Pressure

#     ## Fluid properties
#     mu = sympy.Symbol('mu', positive = True)         # Dynamic viscosity
#     rho = sympy.Symbol('rho', positive = True)       # Density

#     ## Test functions definition
#     w = DefineMatrix('w', v_n_nodes, dim)            # Velocity field test function
#     q = DefineVector('q', p_n_nodes)                 # Pressure field test function

#     ## Other data definitions
#     f = DefineMatrix('r_f',v_n_nodes,dim)                 # Forcing term

#     ## Constitutive matrix definition
#     C = DefineSymmetricMatrix('C',strain_size,strain_size)

#     ## Stress vector definition
#     stress = DefineVector('r_stress',strain_size)

#     ## Other simbols definition
#     h = sympy.Symbol('h', positive = True)                        # Element characteristic size
#     stab_c1 = sympy.Symbol('stab_c1', positive = True)            # Stabilization first constant
#     stab_c2 = sympy.Symbol('stab_c2', positive = True)            # Stabilization second constant
#     dyn_tau = sympy.Symbol('dyn_tau', positive = True)            # Stabilization dynamic tau
#     dt = sympy.Symbol('rData.DeltaTime', positive = True)         # Time increment
#     gauss_weight = sympy.Symbol('gauss_weight', positive = True)  # Integration point weight

#     ## Backward differences coefficients
#     bdf0 = sympy.Symbol('rData.BDF0')
#     bdf1 = sympy.Symbol('rData.BDF1')
#     bdf2 = sympy.Symbol('rData.BDF2')

#     ## Convective velocity definition
#     if linearisation == "Picard":
#         vconv = DefineMatrix('vconv',v_n_nodes,dim)     # Convective velocity defined a symbol
#     elif linearisation == "FullNR":
#         vmesh = DefineMatrix('r_vmesh',v_n_nodes,dim)   # Mesh velocity
#         vconv = v - vmesh                               # Convective velocity defined as a velocity dependent variable
#     else:
#         raise Exception(f"Wrong linearisation \'{linearisation}\' selected. Available options are \'Picard\' and \'FullNR\'.")
#     vconv_gauss = vconv.transpose()*N_v

#     ## Compute the rest of magnitudes at the Gauss points
#     accel_gauss = (bdf0*v + bdf1*vn + bdf2*vnn).transpose()*N_v

#     ## Data interpolation to the Gauss points
#     f_gauss = f.transpose()*N_v

#     v_gauss = v.transpose()*N_v
#     p_gauss = p.transpose()*N_p

#     w_gauss = w.transpose()*N_v
#     q_gauss = q.transpose()*N_p

#     ## Gradients computation (fluid dynamics gradient)
#     grad_w = DfjDxi(DN_v, w)
#     grad_q = DfjDxi(DN_p, q)

#     grad_v = DfjDxi(DN_v,v)
#     grad_p = DfjDxi(DN_p, p)

#     div_w = div(DN_v,w)
#     div_v = div(DN_v,v)

#     div_vconv = div(DN_v, vconv)

#     grad_sym_v = grad_sym_voigtform(DN_v,v)       # Symmetric gradient of v in Voigt notation
#     grad_sym_w_voigt = grad_sym_voigtform(DN_v,w) # Symmetric gradient of w in Voigt notation
#     # Recall that the grad(w):stress contraction equals grad_sym(w)*stress in Voigt notation since the stress is a symmetric tensor.

#     # Convective term definition
#     convective_term_gauss = vconv_gauss.transpose()*grad_v

#     ## Compute galerkin functional
#     # Navier-Stokes functional
#     rv_galerkin = rho*w_gauss.transpose()*f_gauss - rho*w_gauss.transpose()*accel_gauss - grad_sym_w_voigt.transpose()*stress + div_w*p_gauss - q_gauss*div_v
#     rv_galerkin -= rho*w_gauss.transpose()*convective_term_gauss.transpose()

#     ## Stabilization functional
#     stab_norm_a = 0.0
#     for i in range(dim):
#         stab_norm_a += vconv_gauss[i]**2
#     stab_norm_a = sympy.sqrt(stab_norm_a)
#     tau_1 = 1.0/(rho*dyn_tau/dt + stab_c2*rho*stab_norm_a/h + stab_c1*mu/h**2) # Velocity stabilization operator
#     tau_2 = mu + stab_c2*rho*stab_norm_a*h/stab_c1                             # Pressure stabilization operator

#     C_aux = ConvertVoigtMatrixToTensor(C) # Definition of the 4th order constitutive tensor from the previous definition symbols
#     div_stress = sympy.zeros(dim, 1)
#     for i in range(dim):
#         for j in range(dim):
#             for k in range(dim):
#                 for m in range(dim):
#                     for n in range(v_n_nodes):
#                         div_stress[i] += 0.5*C_aux[i,j,k,m]*(DDN_v[0,n][i,m]*v[n,k] + DDN_v[0,n][i,k]*v[n,m])

#     mom_residual = rho*f_gauss - rho*accel_gauss - rho*convective_term_gauss.transpose() + div_stress - grad_p
#     vel_subscale = tau_1 * mom_residual

#     mass_residual = - div_v
#     pres_subscale = tau_2 * mass_residual

#     rv_stab = grad_q.transpose()*vel_subscale
#     rv_stab += rho*vconv_gauss.transpose()*grad_w*vel_subscale
#     rv_stab += rho*div_vconv*w_gauss.transpose()*vel_subscale
#     if add_pressure_subscale:
#         rv_stab += div_w * pres_subscale

#     ## Define DOFs and test function vectors
#     n_dofs = v_n_nodes * dim + p_n_nodes

#     dofs = sympy.zeros(n_dofs, 1)
#     testfunc = sympy.zeros(n_dofs, 1)

#     # Velocity DOFs and test functions
#     for i in range(v_n_nodes):
#         for k in range(dim):
#             dofs[i*dim + k] = v[i,k]
#             testfunc[i*dim + k] = w[i,k]

#     # Pressure DOFs and test functions
#     for i in range(p_n_nodes):
#         dofs[v_n_nodes*dim + i] = p[i,0]
#         testfunc[v_n_nodes*dim + i] = q[i,0]

#     ## Compute LHS and RHS
#     # Add the stabilization to the Galerkin residual
#     functional = rv_galerkin + rv_stab

#     # For the RHS computation one wants the residual of the previous iteration (residual based formulation). By this reason the stress is
#     # included as a symbolic variable, which is assumed to be passed as an argument from the previous iteration database.
#     print(f"Computing {dim}D RHS Gauss point contribution\n")
#     rhs = Compute_RHS(functional.copy(), testfunc, do_simplifications)
#     rhs_out = OutputVector_CollectingFactors(gauss_weight*rhs, "rRHS", mode, assignment_op='+=')

#     # Compute LHS (RHS(residual) differenctiation w.r.t. the DOFs)
#     # Note that the 'stress' (symbolic variable) is substituted by 'C*grad_sym_v' for the LHS differenctiation. Otherwise the velocity terms
#     # within the velocity symmetryc gradient would not be considered in the differenctiation, meaning that the stress would be considered as
#     # a velocity independent constant in the LHS.
#     print(f"Computing {dim}D LHS Gauss point contribution\n")
#     SubstituteMatrixValue(rhs, stress, C*grad_sym_v)
#     lhs = Compute_LHS(rhs, testfunc, dofs, do_simplifications) # Compute the LHS (considering stress as C*(B*v) to derive w.r.t. v)
#     lhs_out = OutputMatrix_CollectingFactors(gauss_weight*lhs, "rLHS", mode, assignment_op='+=')

#     ## Replace the computed RHS and LHS in the template outstring
#     outstring = outstring.replace(f"//substitute_lhs_{dim}D", lhs_out)
#     outstring = outstring.replace(f"//substitute_rhs_{dim}D", rhs_out)

# ## Write the modified template
# print(f"Writing output file \'{output_filename}\'")
# out = open(output_filename,'w')
# out.write(outstring)
# out.close()
