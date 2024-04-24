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