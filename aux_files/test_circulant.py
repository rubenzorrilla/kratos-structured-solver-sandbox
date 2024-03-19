import numpy as np
import scipy
import scipy.linalg
from scipy.linalg import circulant




n = 10
c = np.zeros(n)
c[0] = 2
c[1] = -1
c[-1] = -1
A = circulant(c)

#A = np.zeros((n,n))
b = np.array(np.arange(n))

print(scipy.linalg.solve_circulant(c, b, singular='lstsq'))

# for i in range(n):
#     if(i-1>=0):
#         A[i,i-1] = -1

#     A[i,i] = 2
#     if(i+1<n):
#         A[i,i+1] = -1

#print("reference sol = ", np.linalg.inv(A)@b)


# fft_c = np.fft.fft(c)
# def apply_precond(r):
#     print(np.linalg.norm(r))
#     fft_r = np.fft.fft(r)
#     return np.fft.ifft(fft_r/fft_c)

# print("sol ", apply_precond(b))