import numpy
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

print("Reading pressure_matrix_without_bcs.mm...")
pressure_matrix_without_bcs = scipy.io.mmread('pressure_matrix_without_bcs.mm')
pressure_matrix_without_bcs = pressure_matrix_without_bcs.tocsr()
print("pressure_matrix_without_bcs.mm imported.")

# Case 1: solve the problem with no preconditioner
print("\nCASE 1")
print("\tSolving CG (no preconditioner) problem...")
p_iters = 0
delta_p, converged = scipy.sparse.linalg.cg(pressure_matrix, b, atol=1.0e-7, callback=nonlocal_iterate, M=None)
print(f"\tp_iters: {p_iters}")

# Case 2: use the inverse of the pressure matrix without BCs as preconditioner
print("\nCASE 2")
print("\tComputing pressure matrix without BCs inverse...")
Minv = scipy.sparse.linalg.inv(pressure_matrix_without_bcs.tocsc())

print("\tSolving PCG problem ...")
p_iters = 0
delta_p, converged = scipy.sparse.linalg.cg(pressure_matrix, b, atol=1.0e-7, callback=nonlocal_iterate, M=Minv)
print(f"\tp_iters (w/ preco): {p_iters}")

# Case 3: use the circulant matrix inverse as preconditioner by FFT
print("\nCASE 3")
print("\tComputing FFT linear operator preconditioner...")
x = numpy.zeros((b.shape[0]))
x[55] = 1.0 # For a 10 x 10 mesh, cell 55 is "more or less" in the middle
y = pressure_matrix_without_bcs @ x
fft_x = numpy.fft.fft(x)
fft_y = numpy.fft.fft(y)
fft_c = numpy.real(fft_y / fft_x)# Take the real part only (imaginary one is zero)
fft_c[0] = 1.0e0 # Remove the first coefficient as this is associated to the solution average

def apply_precond(r):
    fft_r = numpy.fft.fft(r)
    return numpy.real(numpy.fft.ifft(fft_r/fft_c))
precond = scipy.sparse.linalg.LinearOperator((b.shape[0], b.shape[0]), matvec=apply_precond)

print("\tSolving PCG problem...")
p_iters = 0
delta_p, converged = scipy.sparse.linalg.cg(pressure_matrix, b, atol=1.0e-7, callback=nonlocal_iterate, M=precond)
print(f"\tp_iters (w/ preco): {p_iters}")

# Case 4: build the circulant pressure matrix from its FFT and use its inverse as preconditioner
print("\nCASE 4")
print("\tBuilding circulant pressure matrix from its FFT...")
C = scipy.linalg.circulant(numpy.real(numpy.fft.ifft(fft_c)))

print("\tComputing circulant pressure matrix inverse...")
Cinv = scipy.linalg.inv(C)

print("\tSolving PCG problem ...")
p_iters = 0
delta_p, converged = scipy.sparse.linalg.cg(pressure_matrix, b, atol=1.0e-7, callback=nonlocal_iterate, M=Cinv)
print(f"\tp_iters (w/ preco): {p_iters}")

# Case 5: use a inverse of the pressure matrix as diagonal preconditioner
print("\nCASE 5")
print("\tComputing pressure matrix inverse...")
Minv = scipy.sparse.linalg.inv(pressure_matrix.tocsc())
print("\tBuilding diagonal preconditioner...")
Minv_diag = scipy.sparse.diags(Minv.diagonal())

print("\tSolving PCG problem ...")
p_iters = 0
delta_p, converged = scipy.sparse.linalg.cg(pressure_matrix, b, atol=1.0e-7, callback=nonlocal_iterate, M=Minv_diag)
print(f"\tp_iters (w/ preco): {p_iters}")

# Case 6: use the pressure matrix inverse as preconditioner (must converge in one iteration)
print("\nCASE 6")
print("\tComputing pressure matrix inverse to be used as preconditioner...")
Minv = scipy.sparse.linalg.inv(pressure_matrix.tocsc())

print("\tSolving PCG problem ...")
p_iters = 0
delta_p, converged = scipy.sparse.linalg.cg(pressure_matrix, b, atol=1.0e-7, callback=nonlocal_iterate, M=Minv)
print(f"\tp_iters (w/ preco): {p_iters} (must be one)")

# Case 7: symmetric successive over-relaxation preconditioning
print("\nCASE 7")
print("\tComputing SSOR preconditioner...")
D = scipy.sparse.diags(pressure_matrix.diagonal())
Dinv = scipy.sparse.linalg.inv(D.tocsc())
L = scipy.sparse.tril(pressure_matrix)
w = 1.0
P = (w/(2-w)) * (D/w + L) @ Dinv @ (D/w + L).transpose()
Pinv = scipy.sparse.linalg.inv(P.tocsc())

print("\tSolving PCG problem ...")
p_iters = 0
delta_p, converged = scipy.sparse.linalg.cg(pressure_matrix, b, atol=1.0e-7, callback=nonlocal_iterate, M=Pinv)
print(f"\tp_iters (w/ preco): {p_iters}")

# Case 8: solving Toeplitz system as preconditioner
print("\nCASE 8")
print("\tComputing Toeplitz preconditioner...")
a = numpy.zeros(pressure_matrix.shape[0])
counter = 0
for i in range(55,pressure_matrix.shape[0]):
    a[counter] = pressure_matrix[55,i]
    counter += 1

def apply_precond(r):
    return scipy.linalg.solve_toeplitz((a,a.copy()),b)
precond = scipy.sparse.linalg.LinearOperator((b.shape[0], b.shape[0]), matvec=apply_precond)

print("\tSolving with Toeplitz ...")
p_iters = 0
delta_p, converged = scipy.sparse.linalg.cg(pressure_matrix, b, atol=1.0e-7, callback=nonlocal_iterate, M=precond)
print(f"\tp_iters (w/ preco): {p_iters}")

# Case 9: solving deflated system
print("\nCASE 9")
print("\tComputing deflation preconditioner...")
Z = numpy.zeros((pressure_matrix.shape[0],3)) # Subspace basis matrix
for i in range(pressure_matrix.shape[0]):
    row = int(i / 10)
    col = i % 10
    Z[i, 0] = 1.0
    Z[i, 1] = 0.1 * col + 0.05
    Z[i, 2] = 0.1 * row + 0.05

E = Z.transpose() @ pressure_matrix @ Z
Einv = scipy.linalg.pinv(E)
P = scipy.sparse.eye(pressure_matrix.shape[0])
P -= pressure_matrix @ Z @ Einv @ Z.transpose()
PA = P @ pressure_matrix
Pb = P@b
PAinv = scipy.linalg.pinv(PA)
y = PAinv @ Pb
lamb = Einv @ (Z.transpose() @ b - Z.transpose() @ pressure_matrix @ y)
delta_p = y + Z @ lamb
print(f"\tDirect solution of the deflated problem error: {scipy.linalg.norm(b.flatten() - (pressure_matrix @ delta_p).flatten())}")

print("\tSolving deflated CG problem ...")
p_iters = 0
y, converged = scipy.sparse.linalg.cg(PA, Pb, atol=1.0e-7, callback=nonlocal_iterate, M=None) #FIXME: the problem is in here!
aux_1 = numpy.array(Z.transpose() @ b)
aux_2 = numpy.array(Z.transpose() @ pressure_matrix @ y)
aux_3 = numpy.empty((Einv.shape[0]))
for i in range(Einv.shape[0]):
    aux_3[i] = aux_1[i,0] - aux_2[i]
lamb = Einv @ aux_3
delta_p = y + Z @ lamb
print(f"\tp_iters (wo/ preco): {p_iters}")
print(f"\tSolution with CG of the deflated problem error: {scipy.linalg.norm(b.flatten() - (pressure_matrix @ delta_p).flatten())}")

print("\tBuilding circulant pressure matrix inverse from its FFT...")
C = scipy.linalg.circulant(numpy.real(numpy.fft.ifft(fft_c)))
Cinv = scipy.linalg.inv(C)

print("\tSolving deflated PCG problem ...")
p_iters = 0
y, converged = scipy.sparse.linalg.cg(PA, Pb, atol=1.0e-7, callback=nonlocal_iterate, M=Cinv) #FIXME: the problem is in here!
aux_1 = numpy.array(Z.transpose() @ b)
aux_2 = numpy.array(Z.transpose() @ pressure_matrix @ y)
aux_3 = numpy.empty((Einv.shape[0]))
for i in range(Einv.shape[0]):
    aux_3[i] = aux_1[i,0] - aux_2[i]
lamb = Einv @ aux_3
delta_p = y + Z @ lamb
print(f"\tp_iters (w/ preco): {p_iters}")
print(f"\tSolution with PCG of the deflated problem error: {scipy.linalg.norm(b.flatten() - (pressure_matrix @ delta_p).flatten())}")
