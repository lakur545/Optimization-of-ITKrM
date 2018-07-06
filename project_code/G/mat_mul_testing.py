#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 10 10:36:48 2018

@author: niels
"""
import numba as nb
import numpy as np
import scipy as sp
import cupy as cp
from numba import cuda
import math
from time import time
import matplotlib.pyplot as plt
np.random.seed(20)
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



M = [1, 5, 1, 1, 5, 10, 25, 50, 100, 250, 500, 750, 1000]
N = [1, 5, 1, 1, 5, 10, 25, 50, 100, 250, 500, 750, 1000]
K = [1, 5, 1, 1, 5, 10, 25, 50, 100, 250, 500, 750, 1000]
time_res = np.zeros((5,len(M)))
for i in range(len(M)):
    A = np.random.rand(M[i],N[i]).reshape(-1)
    B = np.random.rand(N[i],K[i]).reshape(-1)
    res = np.random.rand(M[i],K[i]).reshape(-1)

    threadsperblock = (32, 32)
    blockspergrid_x = math.ceil(M[i] / threadsperblock[0])
    blockspergrid_y = math.ceil(K[i] / threadsperblock[1])
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    
    t0 = time()
    d_A = cuda.to_device(A)
    d_B = cuda.to_device(B)
    d_res = cuda.to_device(res)
    
    t1 = time()
    matmul[threadsperblock, blockspergrid](M[i], N[i], K[i], d_A, d_B, d_res)
    t2 = time()
    
    result_array = d_res.copy_to_host()
    t3 = time()
    
    GPU_no_trans = t2-t1
    GPU_trans = t3-t0
    print("GPU time, no transfer {} seconds".format(GPU_no_trans))
    print("GPU time, transfer {} seconds".format(GPU_trans))
    A = A.reshape(M[i], N[i])
    B = B.reshape(N[i], K[i])
    t0 = time()
    np.dot(A,B)
    np.dot(A,B)
    np.dot(A,B)
    np.dot(A,B)
    np.dot(A,B)
    t1 = time()
    CPU = t1-t0
    print("CPU time {} seconds".format(CPU))
    
    t0 = time()
    GPU_A = cp.asarray(A)
    GPU_B = cp.asarray(A)
    
    t1 = time()
    T = cp.dot(GPU_A, GPU_B)
    t2 = time()
    
    returned_result = cp.asnumpy(T)
    t3 = time()
    cp_GPU_no_trans = t2-t1
    cp_GPU_trans = t3-t0
    print("CuPy GPU time, no transfer {} seconds".format(cp_GPU_no_trans))
    print("CuPy GPU time, transfer {} seconds".format(cp_GPU_trans))
    
    time_res[0, i] = GPU_no_trans
    time_res[1, i] = GPU_trans
    time_res[2, i] = CPU
    time_res[3, i] = cp_GPU_no_trans
    time_res[4, i] = cp_GPU_trans
    



time_res = time_res * 1000

plt.close('all')
plt.figure(1, dpi=200)
#plt.plot(M,time_res[0, :], label = "Custom kernel GPU no transfer")
#plt.plot(M,time_res[1, :], label = "Custom kernel GPU with transfer")
plt.plot(M[3:],time_res[3, 3:], label = "CuPy GPU no transfer")
plt.plot(M[3:],time_res[4, 3:], label = "CuPy GPU with transfer")
plt.plot(M[3:],time_res[2, 3:]/5, label = "CPU")
plt.legend()
plt.xlabel("Dimension [N x N]", fontsize=16)
plt.ylabel("Time [ms]", fontsize=16)
plt.yscale("log")
ax = plt.gca()
plt.setp(ax.get_xticklabels(), fontsize=14)
plt.setp(ax.get_yticklabels(), fontsize=14)
plt.grid(which="major",ls="-", color='grey')
plt.tight_layout()
plt.show()
