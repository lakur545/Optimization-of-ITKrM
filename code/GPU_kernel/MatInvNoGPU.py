# -*- coding: utf-8 -*-
"""
MatInv using LU
"""

import numpy as np
import scipy as sp

def MatInv(N, K, A, x, b, z, L, U):
    for n in range(N):
        # LU decomposition - Crout
        for i in range(0, K):
                for j in range(i, K):
                    L[i + j*K + n*K*K] = A[i + j*K + n*K*K]
                    for k in range(0, i):
                        L[i + j*K + n*K*K] -= L[k + j*K + n*K*K] * U[i + k*K + n*K*K]
    
                U[i + i*K + n*K*K] = 1
                for j in range(i+1, K):
                    U[j + i*K + n*K*K] = A[j + i*K + n*K*K] / L[i + i*K + n*K*K]
                    for k in range(0, i):
                        U[j + i*K + n*K*K] -= (L[k + i*K + n*K*K] * U[j + k*K + n*K*K]) / L[i + i*K + n*K*K]
        
        # Solve system
        for i in range(K):
            z[i + n*K] = b[i + n*K] / L[i + i*K + n*K*K]
            for j in range(0, i):
                z[i + n*K] -= (L[j + i*K + n*K*K] * z[j + n*K]) / L[i + i*K + n*K*K]
        for i in range(K-1, 0-1, -1):
            x[i + n*K] = z[i + n*K]
            for j in range(i+1, K):
                x[i + n*K] -= U[j + i*K + n*K*K] * x[j + n*K]
    
    return x, L, U


B = np.array([[1,4,4 , 0,1,1 , 3,3,0, 3,2,1.1 , 2,6,1 , 1,4,2]]).reshape(2,3,3)
A = B[1,:,:]*1
Ainv = np.linalg.inv(A)
b = np.array([[95,22,48],[15.1,29,20]])

K = A.shape[0]
P, L, U = sp.linalg.lu(A)
A2 = B.reshape(-1)*1
Ainv2 = np.zeros((A.shape)).reshape(-1)*1
TpB = 32
BpG = 1
N = 2
b2 = b.reshape(-1)*1
z2 = np.zeros(N*K)
x2 = np.zeros(N*K)
L2 = np.zeros(N*K*K).reshape(-1)*1
U2 = np.zeros(N*K*K).reshape(-1)*1
x, L2, U2 = MatInv(N, K, A2, x2, b2, z2, L2, U2)



