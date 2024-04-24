import numpy
import scipy.io
import scipy.sparse.linalg

def nonlocal_iterate(arr):
    global p_iters
    p_iters += 1

print("Reading b.mm...")
b = scipy.io.mmread('b.mm')
b = b.toarray()
print("b.mm imported.")

print("Reading pressure_matrix.mm...")
pressure_matrix = scipy.io.mmread('pressure_matrix.mm')
pressure_matrix = pressure_matrix.tocsr()
print("pressure_matrix.mm imported.")

print("Reading pressure_matrix_periodic.mm...")
pressure_matrix_periodic = scipy.io.mmread('pressure_matrix_periodic.mm')
pressure_matrix_periodic = pressure_matrix_periodic.tocsr()
print("pressure_matrix_periodic.mm imported.")

print("CASE 1")
print("\tSolving CG (no preconditioner) problem...")
p_iters = 0
delta_p, converged = scipy.sparse.linalg.cg(pressure_matrix, b, atol=1.0e-7, callback=nonlocal_iterate, M=None)
print(f"\tp_iters: {p_iters}")

print("CASE 2")
print("\tComputing periodic pressure matrix inverse...")
Minv = scipy.sparse.linalg.inv(pressure_matrix_periodic.tocsc())

print("\tSolving PCG problem ...")
p_iters = 0
delta_p, converged = scipy.sparse.linalg.cg(pressure_matrix, b, atol=1.0e-7, callback=nonlocal_iterate, M=Minv)
print(f"\tp_iters (w/ preco): {p_iters}")

print("CASE 3")
print("\tComputing FFT linear operator preconditioner...")
x = numpy.zeros((b.shape[0]))
x[55] = 1.0 # For a 10 x 10 mesh, cell 55 is "more or less" in the middle
y = pressure_matrix_periodic @ x
fft_x = numpy.fft.fft(x)
fft_y = numpy.fft.fft(y)
fft_c = numpy.real(fft_y / fft_x)# Take the real part only (imaginary one is zero)
fft_c[0] = 1.0 # Remove the first coefficient as this is associated to the solution average

def apply_precond(r):
    fft_r = numpy.fft.fft(r)
    return numpy.real(numpy.fft.ifft(fft_r/fft_c))

precond = scipy.sparse.linalg.LinearOperator((b.shape[0], b.shape[0]), matvec=apply_precond)

print("\tSolving PCG problem...")
p_iters = 0
Minv = scipy.sparse.linalg.inv(pressure_matrix_periodic.tocsc())
delta_p, converged = scipy.sparse.linalg.cg(pressure_matrix, b, atol=1.0e-7, callback=nonlocal_iterate, M=Minv)
print(f"\tp_iters (w/ preco): {p_iters}")
