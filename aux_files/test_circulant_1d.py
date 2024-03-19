import numpy as np
import numpy.fft
import scipy
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg
import inspect

n = 5000

#circulant stencil
c = np.zeros(n)
c[0] = 2.0
c[1] = -1.0
c[-1] = -1.0

#rhs
b = np.zeros(n)
b[0] = 5.0
b[-1] = 10.0

#matrix with BCs
A = scipy.sparse.diags([-1, 2, -1], [-1, 0, 1], shape=(n,n)).tocsr()


#fft of circulant stencil
## The first coefficient is null, because the Laplacian is not
## PD, just SPD. But we can replace this null coefficient by anything
## different from 0. At most it would degrade the convergence of the
## PCG, but we will see that the convergence is OK.
Cf = np.real(np.fft.fft(c))
Cf[0] = 1.0e-12

def precofunction(y):
    global Cf
    x = np.fft.ifft(np.fft.fft(y)/Cf)
    x = np.real(x)
    return x
P = scipy.sparse.linalg.LinearOperator(A.shape, matvec=precofunction)

def report(xk):
    frame = inspect.currentframe().f_back
    print(frame.f_locals['iter_'],frame.f_locals['resid'])

iters=0
print("Preconditioned")
x,info = scipy.sparse.linalg.cg(A,b,tol=1e-10,callback=report,M=P)
# print("Without preconditioner")
# x,info = scipy.sparse.linalg.cg(A,b,tol=1e-10,callback=report)