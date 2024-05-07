import numpy
import pyamg
import scipy.io
import scipy.linalg
import scipy.sparse.linalg

# Auxiliary callback function for Scipy CG solver
def nonlocal_iterate(arr):
    global p_iters
    p_iters += 1

# Parsing pressure matrices and right hand side
print("Reading b.mm...")
b = scipy.io.mmread('b.mm')
b = b.toarray()
print("b.mm imported.")

print("Reading pressure_matrix.mm...")
pressure_matrix = scipy.io.mmread('pressure_matrix.mm')
pressure_matrix = pressure_matrix.tocsr()
print("pressure_matrix.mm imported.")

# Case 1: solve the problem with no preconditioner
print("\nCASE 1")
print("\tSolving CG (no preconditioner) problem...")
p_iters = 0
delta_p, converged = scipy.sparse.linalg.cg(pressure_matrix, b, atol=1.0e-7, callback=nonlocal_iterate, M=None)
print(f"\tp_iters: {p_iters}")

ml = pyamg.ruge_stuben_solver(pressure_matrix)                    # construct the multigrid hierarchy
print(ml)                                           # print hierarchy information
x = ml.solve(b, tol=1e-7)                          # solve Ax=b to a tolerance of 1e-10
print("residual: ", numpy.linalg.norm(b - pressure_matrix*x))          # compute norm of residual vector
