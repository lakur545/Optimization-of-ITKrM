# -*- coding: utf-8 -*-
"""
Created on Wed Apr  4 08:33:32 2018

@author: Niels
"""
import math
import os
from time import time
import numpy as np
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import skcuda.linalg as linalg
from numba import cuda
if os.system("cl.exe"):
    os.environ['PATH'] += ';'+r"C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\bin"
if os.system("cl.exe"):
    raise RuntimeError("cl.exe still not found, path probably incorrect")


linalg.init()

@cuda.jit
def GPU_pinv(A):
    return np.linalg.pinv(A)


@cuda.jit
def max_atoms(D, yn, S):
    all_abs = abs(linalg.dot(D.T, yn))
    indices = np.argpartition(all_abs, -S)[-S:]
    return indices


D = np.random.rand(256, 200)

# CUDA kernel


@cuda.jit
def foo(C):
    row = cuda.grid(1)
    if row >= C.shape[0]:
        return
    for i in range(C.shape[1]):
        C[row, i] = 2
#    C[row, col] = 2
# Host code
# Allocate memory on the device for the result


C_GPU = cuda.device_array((200, 256))

# Configure the blocks
THREADS_PER_BLOCK = (32, 32)
BLOCKS_PER_GRID_X = int(math.ceil(C_GPU.shape[0] / THREADS_PER_BLOCK[0]))
BLOCKS_PER_GRID_Y = int(math.ceil(C_GPU.shape[1] / THREADS_PER_BLOCK[1]))
BLOCKS_PER_GRID = (BLOCKS_PER_GRID_X, BLOCKS_PER_GRID_Y)

# Start the kernel
START = time()
foo[BLOCKS_PER_GRID, THREADS_PER_BLOCK](C_GPU)
print(time() - START)
# Copy the result back to the host
C = C_GPU.copy_to_host()

print(C)

A = np.random.rand(2, 2)
CU_A = cuda.device_array_like(A)
PYCU_A = pycuda.gpuarray.GPUArray(
    shape=CU_A.shape, dtype=CU_A.dtype,
    gpudata=CU_A.gpu_data.device_ctypes_pointer.value,
    strides=CU_A.strides)
PYCU_A.get()

X = np.asarray(np.random.rand(57, 57), np.float32)
Y = np.asarray(np.random.rand(4, 4), np.float32)
X_GPU = gpuarray.to_gpu(X)
Y_GPU = gpuarray.to_gpu(Y)


START = time()
Z_GPU = linalg.pinv(PYCU_A, lib='cusolver')
GPU_RES = Z_GPU.get()
print(time() - START)

START = time()
CPU_RES = np.linalg.pinv(A)
print(time() - START)
