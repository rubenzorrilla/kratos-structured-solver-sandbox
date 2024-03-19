import math
import sympy

def SetNodalCoordinates(x0, y0, z0, a, b, c):
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

def ShapeFunctionsSecondGradients(nodal_coordinates, gauss_coordinates, jacobian):
    hessian = CalculateHessian(gauss_coordinates, nodal_coordinates)
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
