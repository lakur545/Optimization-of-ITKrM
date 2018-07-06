"""
GPU code testing. Numba
"""

import numba as nb
import numpy as np
import scipy as sp
from numba import cuda
import math
from time import time

@cuda.jit
def increment_by_one(an_array):
    # Thread id in a 1D block
    tx = cuda.threadIdx.x
    # Block id in a 1D grid
    ty = cuda.blockIdx.x
    # Block width, i.e. number of threads per block
    bw = cuda.blockDim.x
    # Compute flattened index inside the array
    pos = tx + ty * bw
    if pos < an_array.size:  # Check array boundaries
        an_array[pos] += 1

@cuda.jit
def saxpy(n, a, x, y, res):
    """
    res = ax+y
    """
    stride = cuda.blockIdx.x * cuda.blockDim.x
    idx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
#    idy = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
#    for i in range(math.ceil(n/stride)):
#        res[idx + i*stride] = a*x[idx + i*stride] + y[idx + i*stride]
    for i in range(1000):
        res[idx] = a*x[idx]*x[idx] + y[idx] * y[idx]
#    res[idx] = cuda.blockDim.x

@cuda.jit
def matmul(M, N, K, A, B, res):
    col = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x  # The column of the resulting matrix [0, K]
    row = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y  # The row of the resulting matrix [0, M]
    lay = cuda.blockIdx.z * cuda.blockDim.z + cuda.threadIdx.z  # The layer of the resulting matrix (if 3d matrix)
    if (row < M and col < K):
        tmp_sum = 0
        for i in range(N):
            tmp_sum += A[row * N + i] * B[i * K + col]
        res[row * K + col] = tmp_sum

@cuda.jit
def matinv(N, K, A, x, b, z, L, U):
    n = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    # 2d array( A[i][j] )= 1d array( A[j + i*K] )
    # if n < N:
    #     for i in range(0, K):
    #         for k in range(i, K):
    #             tmp_sum = 0
    #             for j in range(0, i):
    #                 tmp_sum += L[j + i*K + n*K*K] * U[k + j*K + n*K*K]
    #             U[k + i*K + n*K*K] = A[k + i*K] - tmp_sum

    #         for k in range(i, K):
    #             if (i == k):
    #                 L[i+ i*K + n*K*K] = 1
    #             else:
    #                 tmp_sum = 0
    #                 for j in range(0, i):
    #                     tmp_sum += L[j + k*K + n*K*K] * U[i + j*K + n*K*K]
    #                 L[i + k*K + n*K*K] = (A[i + k*K] - tmp_sum) / U[i + i*K + n*K*K]

    if n < N:
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


@cuda.jit
def D_kernel_simple(M, N, K, S, D, DtD, I_D, Y, vecproj):
    n = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    ### Simple version of 1 training example
    if (n < N):
        # Vector projection
        for m in range(M):
            # Calculate D_((I_D)_n).T @ Y_n
            tmp_sum = 0
            for i in range(M):
                tmp_sum += D[i + I_D[m] * M] * Y[i]
            # Calculate D*1/diag(DtD)*DtY
            for s in range(S):
                vecproj[m + s * M] = D[s + I_D[m] * M] / DtD[I_D[m] + I_D[m] * K] * tmp_sum

@cuda.jit
def D_kernel(M, N, K, S, D, DtD, I_D, Y, vecproj, test):
    """
    Original shapes:
        M, K = D
        K, K = DtD
        S, N = I_D
        M, N = Y
        M, S = vecproj
        M, S = D_((I_D)_n)

        M, 1 = Y_n
        M, 1 = D_((I_D)_n).T @ Y_n
    """
#    col = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x  # The column of the resulting matrix [0, K]
#    row = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y  # The row of the resulting matrix [0, M]
#    lay = cuda.blockIdx.z * cuda.blockDim.z + cuda.threadIdx.z  # The layer of the resulting matrix (if 3d matrix)
    n = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    ### Simple version of 1 training example
    
    if (n < N):
        # Vector projection
        # Calculate D_((I_D)_n).T @ Y_n
        for s in range(S):
            tmp_sum = 0
            for i in range(M):
                tmp_sum += D[I_D[n + s * N] + i * K] * Y[n + i * N]
#            test[n + s * N] = tmp_sum   # DtY

            # Calculate 1/diag(DtD)
#            test[n + s * N] = 1/DtD[I_D[n + s * N] + I_D[n + s * N] * K]
            
            # Calculate D/diag(DtD)*DtY
            for m in range(M):
                vecproj[s + m * S + n * M * S] = D[I_D[n + s * N] + m * K]*tmp_sum/DtD[I_D[n + s * N] + I_D[n + s * N] * K]
            
            # s*M+m+n*M*S is the s-th row and m-th column of the n-th training example
            # I_D[m+n*S] + I_D[m+n*S] * K takes the diagonal elements of DtD using the n-th index set.

### itkrm very small test
# D = np.array([[3.0,2,1],[2,1,1]])
# I_D = np.array([[0,2,1],[1,0,2]])
# DtD = D.T@D
# Y = np.array([[0.1,0.2,0.5],[0.3,0.2,0.1]])
# M, K = D.shape
# S, N = I_D.shape
# n = 0
# DtY = D[:,I_D[:,n]].T@Y[:,n]
# vecproj = D[:,I_D[:,n]] @ np.diag(np.diag( DtD[I_D[:,n,None], I_D[:,n]] )**-1*( DtY ))

#D2 = D.T.reshape(-1)*1
#DtD2 = DtD.reshape(-1)*1
#I_D2 = I_D.T.reshape(-1)*1
#Y2 = Y.T.reshape(-1)*1
#vecproj2 = np.zeros((M, S)).reshape(-1)*1
#TpB = (1, 1, 1)    # ThreadsPerBlock
#BpG = (1, 1, 1)     # BlocksPerGrid
#D_kernel_simple[TpB, BpG](M, N, K, S, D2, DtD2, I_D2, Y2, vecproj2)
#res = vecproj2.reshape(M, S)

# D3 = D.reshape(-1)*1
# DtD3 = DtD.reshape(-1)*1
# I_D3 = I_D.reshape(-1)*1
# Y3 = Y.reshape(-1)*1
# vecproj3 = np.zeros((N, M, S)).reshape(-1)*1
# test = np.zeros(S*N*M)

# TpB = 32    # ThreadsPerBlock
# BpG = 1     # BlocksPerGrid
# D_kernel[TpB, BpG](M, N, K, S, D3, DtD3, I_D3, Y3, vecproj3, test)

### Matrix inversion
B = np.array([[1,4,4 , 0,1,1 , 3,3,0, 3,2,1.1 , 2,6,1 , 1,4,2]]).reshape(2,3,3)
A = B[1,:,:]*1
Ainv = np.linalg.inv(A)
b = np.array([[95,22,48],[15.1,29,20]])
K = A.shape[0]
#P, L, U = sp.linalg.lu(A)
A2 = B.reshape(-1)*1
Ainv2 = np.zeros((A.shape)).reshape(-1*1)
TpB = 32
BpG = 1
N = 2
b2 = b.reshape(-1)*1
z2 = np.zeros(N*K).reshape(-1)*1
x2 = np.zeros(N*K).reshape(-1)*1
L2 = np.zeros(N*K*K).reshape(-1)*1
U2 = np.zeros(N*K*K).reshape(-1)*1
matinv[TpB, BpG](N, K, A2, x2, b2, z2, L2, U2)

### general testing
#dim = 1000
#an_array = np.arange(dim)
#
#threadsperblock = 32    # Should be in multiple of 32
#blockspergrid = (an_array.size + (threadsperblock - 1)) // threadsperblock

# increment_by_one[blockspergrid, threadsperblock](an_array)

# a = np.arange(dim)
# b = np.arange(dim)
# res = np.zeros(len(a))
# nb.cuda.profile_start()
# saxpy[blockspergrid, threadsperblock](len(a), 2, a, b, res)
# nb.cuda.profile_stop()
# print(res)

M = 1000
N = 100000
K = 1000
A = np.arange(M*N)
B = np.arange(N*K)
res = np.zeros(M*K)

threadsperblock = (32, 32)
blockspergrid_x = math.ceil(M / threadsperblock[0])
blockspergrid_y = math.ceil(K / threadsperblock[1])
blockspergrid = (blockspergrid_x, blockspergrid_y)

t0 = time()
d_A = cuda.to_device(A)
d_B = cuda.to_device(B)
d_res = cuda.to_device(res)

t1 = time()
matmul[threadsperblock, blockspergrid](M, N, K, d_A, d_B, d_res)
t2 = time()

result_array = d_res.copy_to_host()
t3 = time()