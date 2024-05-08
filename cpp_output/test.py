import numpy as np


D = np.array([[-0.5,0.5,0.5,-0.5],[-0.5,-0.5,0.5,0.5]])
D = D.reshape((1,8),order="F")
G = D.T

Minv_2 = np.diag([2.,2.,4.,4.,2.,2.,1.,1.])

Minv_4 = np.diag([1.,1.,1.,1.,1.,1.,1.,1.])

print("D=",D)
print("G=",G)

D2 = D@Minv_2
D4 = D@Minv_4

G4 = np.array([0.5,0.5,  -0.5,0.5,   -0.5,-0.5,   0.5,-0.5]).T
print("D4@G4 " ,D4@G4)

G2 = np.array([0.5,-0.5,  -0.0,0.0,   -0.0,-0.0,   0.5,-0.5]).T
print("D2@G2 " ,D2@G2)