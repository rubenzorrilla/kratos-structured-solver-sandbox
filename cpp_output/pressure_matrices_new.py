import numpy
import pyamg
import scipy.io
import scipy.linalg
import scipy.sparse.linalg
import matplotlib.pyplot as plt

# Auxiliary callback function for Scipy CG solver
def nonlocal_iterate(arr):
    global p_iters
    p_iters += 1

# Mesh size
x_divisions = 128
y_divisions = 128

# Parsing pressure matrices and right hand side
b_name = f'b_{x_divisions}_{y_divisions}_1.mm'
print(f"\nReading {b_name}...")
b = scipy.io.mmread(b_name)
b = b.toarray()
print(f"{b_name} imported.")
print(f"b norm: {numpy.linalg.norm(b)}")

pressure_matrix_name = f'pressure_matrix_{x_divisions}_{y_divisions}.mm'
print(f"\nReading {pressure_matrix_name}...")
pressure_matrix = scipy.io.mmread(pressure_matrix_name)
pressure_matrix = pressure_matrix.tocsr()
print(f"{pressure_matrix_name} imported.")

pressure_matrix_wo_bcs_name = f'pressure_matrix_without_bcs_{x_divisions}_{y_divisions}.mm'
print(f"\nReading {pressure_matrix_wo_bcs_name}...")
pressure_matrix_wo_bcs = scipy.io.mmread(pressure_matrix_wo_bcs_name)
pressure_matrix_wo_bcs = pressure_matrix_wo_bcs.tocsr()
print(f"{pressure_matrix_wo_bcs_name} imported.")

# b_name_py = f'b_{x_divisions}_{y_divisions}_1_py.mtx'
# print(f"\nReading {b_name_py}...")
# b_py = scipy.io.mmread(b_name_py)
# b_py = b_py.transpose()
# print(f"{b_name_py} imported.")
# print(f"b_py norm: {numpy.linalg.norm(b_py)}")
# print(f"Python vs. ccp RHS (nonzeros): {numpy.count_nonzero(b - b_py)}")

# pressure_matrix_name_py = f'pressure_matrix_{8}_{8}_py.mtx'
# print(f"\nReading {pressure_matrix_name_py}...")
# pressure_matrix_py = scipy.sparse.csr_matrix(scipy.io.mmread(pressure_matrix_name_py))
# print(f"{pressure_matrix_name_py} imported.")
# print(f"Python vs. ccp pressure matrix (nonzeros): {(pressure_matrix - pressure_matrix_py).count_nonzero()}")

# plt.spy(pressure_matrix)
# plt.show()

# plt.imshow(pressure_matrix.toarray(), interpolation='antialiased', cmap='binary')
# plt.colorbar()
# plt.show()

# print(pressure_matrix)

# plt.matshow(pressure_matrix.toarray())
# plt.colorbar()
# plt.show()

# plt.matshow(pressure_matrix_wo_bcs.toarray())
# plt.colorbar()
# plt.show()

ones_vect = numpy.ones(pressure_matrix.shape[0])
print(f"\nP @ ones_vect is all zeros? {not any(pressure_matrix@ones_vect)}")
print(f"\nP (periodic) @ ones_vect is all zeros? {not any(pressure_matrix_wo_bcs@ones_vect)}")
print(pressure_matrix_wo_bcs)
print(pressure_matrix_wo_bcs@ones_vect)

# Compute pressure matrix eigenvalues
solve = scipy.sparse.linalg.factorized(pressure_matrix.tocsc())
p_eigvals, _ = scipy.sparse.linalg.eigs(pressure_matrix, which="SM", )
print(f"\nPRESSURE MATRIX EIGENVALUES")
print(f"p_eigvals: {p_eigvals}")
print(f"All positive: {all([val > 0.0 for val in p_eigvals])}")

# Case 1: solve the problem with no preconditioner
print("\nCASE 1")
print("Solving CG (no preconditioner) problem...")
p_iters = 0
delta_p, converged = scipy.sparse.linalg.cg(pressure_matrix, b, atol=1.0e-7, callback=nonlocal_iterate, M=None)
print(f"p_iters: {p_iters}")
print("residual cg: ", numpy.linalg.norm(b.flatten() - pressure_matrix@(delta_p.flatten()))) # compute norm of residual vector

# # Case 1 (py): solve the problem with no preconditioner
# print("\nCASE 1 (py)")
# print("Solving CG (no preconditioner) problem...")
# p_iters = 0
# delta_p_py, converged = scipy.sparse.linalg.cg(pressure_matrix_py, b_py.flatten(), atol=1.0e-7, callback=nonlocal_iterate, M=None)
# print(f"p_iters: {p_iters}")
# print("residual cg: ", numpy.linalg.norm(b_py.flatten() - pressure_matrix_py@(delta_p_py.flatten()))) # compute norm of residual vector

# Case 2: solve the problem with AMG solver
print("\nCASE 2")
print("Solving AMG problem...")
standalone_residuals = []
ml = pyamg.ruge_stuben_solver(pressure_matrix) # construct the multigrid hierarchy
#print(ml) # print hierarchy information
delta_p_ml = ml.solve(b, tol=1.0e-7, residuals=standalone_residuals) # solve Ax=b to a tolerance of 1e-7
print(standalone_residuals)
print("residual amg: ", numpy.linalg.norm(b.flatten() - pressure_matrix@(delta_p_ml.flatten()))) # compute norm of residual vector

# Case 3: use the circulant matrix inverse as preconditioner by FFT
print("CASE 3")
print("Computing FFT linear operator preconditioner...")
x = numpy.zeros((b.shape[0]))
x[150] = 1.0
y = pressure_matrix @ x
fft_x = numpy.fft.fft(x)
fft_y = numpy.fft.fft(y)
# fft_c = numpy.real(fft_y / fft_x)# Take the real part only (imaginary one is zero)
fft_c = fft_y / fft_x# Take the real part only (imaginary one is zero)
fft_c[0] = 1.0e0 # Remove the first coefficient as this is associated to the solution average

def apply_precond(r):
    fft_r = numpy.fft.fft(r)
    return numpy.real(numpy.fft.ifft(fft_r/fft_c))
precond = scipy.sparse.linalg.LinearOperator((b.shape[0], b.shape[0]), matvec=apply_precond)

print("Solving PCG problem...")
p_iters = 0
delta_p_prec, converged = scipy.sparse.linalg.cg(pressure_matrix, b, atol=1.0e-7, callback=nonlocal_iterate, M=precond)
print(f"p_iters (w/ preco): {p_iters}")
print("residual cg: ", numpy.linalg.norm(b.flatten() - pressure_matrix@(delta_p_prec.flatten()))) # compute norm of residual vector

print("CG vs. AMG error: ", numpy.linalg.norm(delta_p_prec - delta_p_ml)) # compute norm of residual vector
