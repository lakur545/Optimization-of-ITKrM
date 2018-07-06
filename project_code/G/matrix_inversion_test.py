# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 14:40:29 2018

@author: Niels
"""
import numpy as np
import cupy as cp
from numba import cuda
from time import time
import matplotlib.pyplot as plt
import math

@cuda.jit("void(float64[:,:],float64[:,:],float64[:,:])")
def matmul(A, B, C):
    """Perform square matrix multiplication of C = A * B
    """
    i, j = cuda.grid(2)
    if i < C.shape[0] and j < C.shape[1]:
        tmp = 0.
        for k in range(A.shape[1]):
            tmp += A[i, k] * B[k, j]
        C[i, j] = tmp


if __name__ == "__main__":
    work = 1
    N = [1, 5, 1, 1, 5, 10, 50, 100, 250, 500, 1000, 2000, 3000]
    T_time = np.zeros((3, len(N)))
    loops = np.ones(len(N),dtype=np.int32)
    
    loops_GPU = np.ones(len(N),dtype=np.int32)
    
    filename="CPU_vs_GPU_transfer_no_transfer_{}.pckl".format(work)
    
    for j in range(len(N)):
        np.random.seed(10)
        AH = np.random.rand(N[j], N[j], loops[j])
        np.ascontiguousarray(AH)
        
        BH = np.random.rand(N[j], N[j], loops[j])
        np.ascontiguousarray(BH)
        
        CPU_time = time()
        TE = np.linalg.inv(AH[:, :, 0])
        
        part_cal = time() - CPU_time
        print("CPU time: {} N: {}".format(part_cal, N[j]))
        T_time[0, j] = part_cal/loops[j]
        
        
        GPU_time = time()

        GPU_A = cp.asarray(AH[:, :, 0])
        
        GPU_time_no_trans = time()

        GPU_TE = (cp.linalg.inv(GPU_A))
        GPU_time_no_trans = time() - GPU_time_no_trans
        
        GPU_res = cp.asnumpy(GPU_TE)

        part_cal = time() - GPU_time
        print("GPU time, transfer: {} N: {}".format(part_cal, N[j]))
        print("GPU time, no transfer: {} N: {}".format(GPU_time_no_trans, N[j]))
        T_time[1, j] = part_cal
        T_time[2, j] = GPU_time_no_trans
    
    plt.close('all')
    
    plt.figure(4, dpi=200)
    
    plt.plot(N[3:], 10**3*T_time[0, 3:], label="CPU time", linewidth=2.5)
    plt.yscale('log')
    
    plt.plot(N[3:], 10**3*T_time[1, 3:], label="CuPy GPU time, with transfer", linewidth=2.5)
    plt.yscale('log')
    
    plt.plot(N[3:], 10**3*T_time[2, 3:], label="CuPy GPU time, no transfer", linewidth=2.5)
    plt.yscale('log')
    
    plt.legend()    
    
    plt.xlabel("Dimension [N x N]", fontsize=16)
    
    plt.ylabel("Time [ms]", fontsize=16)
    
    ax = plt.gca()
    plt.setp(ax.get_xticklabels(), fontsize=14)
    plt.setp(ax.get_yticklabels(), fontsize=14)
    plt.grid(which="major",ls="-", color='grey')
    plt.tight_layout()
    
    plt.show()

